using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using Debug = UnityEngine.Debug;

public class MujocoBridge : MonoBehaviour
{
    [Header("Red")]
    public int statePort = 5005;
    public int cmdPort = 5006;
    public string pythonHost = "127.0.0.1";

    [Header("Botones")]
    public Key clutchKey = Key.Space;
    public Key jawOpenKey = Key.E;
    public Key jawCloseKey = Key.D;
    public Key resetKey = Key.R;

    [Header("Indicador visual (opcional)")]
    public Renderer[] highlightRenderers;
    public Color okColor = new Color(0.2f, 0.8f, 0.3f);
    public Color warnColor = new Color(1f, 0.75f, 0.1f);
    public Color badColor = new Color(0.9f, 0.15f, 0.15f);
    public Color clutchColor = new Color(0.4f, 0.6f, 1f);
    [Range(0f, 1f)] public float warnThreshold = 0.35f;
    [Range(0f, 1f)] public float badThreshold = 0.10f;
    public float jawForceLimitN = 5f;

    [Header("Contactos")]
    public bool showContacts = true;
    public float markerSize = 0.0012f;
    public float forceForMaxSize = 10f;
    public Color jawJawColor = new Color(0.1f, 0.8f, 1f);
    public Color jawObjectColor = new Color(1f, 0.1f, 0.6f);
    public Color otherContactColor = new Color(1f, 0.9f, 0.1f);

    [Header("Maestro (opcional)")]
    public UR3Controller masterView;

    [Header("Log CSV")]
    public bool logCsv = true;
    public string logDir = "";

    [Header("Estado (solo lectura)")]
    public bool connected, masterOk, clutch, calibrated;
    public float simTime, wristSing = 1f, elbowSing = 1f, shoulderSing = 1f, pitchMargin = 1f, jawForce;
    public int contacts;
    public long lastSeq, received, lost, stale, outOfOrder;
    public float latencyMs, holdMs;
    public float[] toolQ = new float[4];

    class Packet
    {
        public JObject st;
        public long seq;
        public double tSend, tRecv;
    }

    UdpClient rx, tx;
    Thread rxThread;
    Packet latest;
    long recvSeq;
    float lastPacketTime = -10f;
    MaterialPropertyBlock mpb;

    bool logging;
    StreamWriter recvLog, frameLog;
    readonly ConcurrentQueue<string> recvRows = new ConcurrentQueue<string>();

    class Link
    {
        public Transform t;
        public Vector3 restPos;   public Quaternion restRot;
        public Vector3 offPos;    public Quaternion offRot;
    }
    readonly Dictionary<string, Link> links = new Dictionary<string, Link>();
    readonly List<string> order = new List<string>();
    readonly List<Renderer> markers = new List<Renderer>();
    readonly Dictionary<string, Transform> prims = new Dictionary<string, Transform>();
    bool bound;

    // same clock as Python's time.perf_counter() on Windows (QueryPerformanceCounter)
    static double Now() => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;

    void Start()
    {
        mpb = new MaterialPropertyBlock();
        if (logCsv) OpenLogs();
        tx = new UdpClient();
        rx = new UdpClient(statePort);
        rxThread = new Thread(ReceiveLoop) { IsBackground = true };
        rxThread.Start();
    }

    void OpenLogs()
    {
        string dir = string.IsNullOrEmpty(logDir) ? Path.Combine(Application.dataPath, "..", "BridgeLogs") : logDir;
        Directory.CreateDirectory(dir);
        string stamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        recvLog = new StreamWriter(Path.Combine(dir, $"unity_recv_{stamp}.csv"));
        frameLog = new StreamWriter(Path.Combine(dir, $"unity_frames_{stamp}.csv"));
        recvLog.WriteLine("seq,t_send,t_recv,bytes");
        frameLog.WriteLine("frame,seq,t_send,t_recv,t_apply,unity_dt_ms,stale_total,lost_total,ncontacts,roll,pitch,jaw_left,jaw_right");
        logging = true;
        Debug.Log("[MujocoBridge] log -> " + Path.GetFullPath(dir) + "  (" + stamp + ")");
    }

    // parses off the main thread; Update only picks up the newest packet
    void ReceiveLoop()
    {
        var ep = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            byte[] raw;
            try { raw = rx.Receive(ref ep); }
            catch (SocketException) { return; }
            catch (System.ObjectDisposedException) { return; }
            double t = Now();

            JObject st;
            try { st = JObject.Parse(Encoding.UTF8.GetString(raw)); }
            catch (Newtonsoft.Json.JsonException) { continue; }
            long seq = (long?)st["seq"] ?? 0;
            double tSend = (double?)st["t_send"] ?? 0;
            received++;

            if (seq <= recvSeq && recvSeq - seq < 100) { outOfOrder++; continue; }
            if (seq > recvSeq + 1 && recvSeq > 0) lost += seq - recvSeq - 1;
            recvSeq = seq;   // a big backwards jump means Python restarted

            if (logging) recvRows.Enqueue(string.Format(CultureInfo.InvariantCulture, "{0},{1:F6},{2:F6},{3}", seq, tSend, t, raw.Length));
            var prev = Interlocked.Exchange(ref latest, new Packet { st = st, seq = seq, tSend = tSend, tRecv = t });
            if (prev != null) stale++;   // overwritten before any frame showed it
        }
    }

    void Update()
    {
        ApplyState();
        SendButtons();
        connected = Time.time - lastPacketTime < 0.5f;
        if (logging)
        {
            while (recvRows.TryDequeue(out string row)) recvLog.WriteLine(row);
            if (Time.frameCount % 60 == 0) { recvLog.Flush(); frameLog.Flush(); }
        }
    }

    void SendButtons()
    {
        var kb = Keyboard.current;
        bool Down(Key k) => kb != null && kb[k].isPressed;
        bool reset = kb != null && kb[resetKey].wasPressedThisFrame;
        string json = string.Format(CultureInfo.InvariantCulture,
            "{{\"buttons\":{{\"clutch\":{0},\"jaw_open\":{1},\"jaw_close\":{2}}},\"ack\":{3},\"hold_ms\":{4:F3}{5}}}",
            B(Down(clutchKey)), B(Down(jawOpenKey)), B(Down(jawCloseKey)), lastSeq, holdMs,
            reset ? ",\"reset\":true" : "");
        byte[] b = Encoding.UTF8.GetBytes(json);
        tx.Send(b, b.Length, pythonHost, cmdPort);
    }

    static string B(bool v) => v ? "true" : "false";

    static string ShortName(string n)
    {
        int i = n.IndexOf("_Shape");
        return i > 0 ? n.Substring(0, i) : n;
    }

    void Bind(JObject bodiesJson)
    {
        var all = GetComponentsInChildren<Transform>(true);
        var found = new List<string>(); var missing = new List<string>();
        foreach (var kv in bodiesJson)
        {
            string mj = kv.Key, key = ShortName(mj);
            Transform hit = null;
            foreach (var t in all) if (t != transform && t.name == mj) { hit = t; break; }
            if (hit == null) foreach (var t in all) if (t != transform && t.name == key) { hit = t; break; }
            if (hit == null) foreach (var t in all) if (t != transform && t.name.StartsWith(key)) { hit = t; break; }
            order.Add(mj);
            if (hit == null) { missing.Add(key); continue; }
            links[mj] = new Link { t = hit };
            found.Add($"{key} → {hit.name}");
        }
        bound = true;
        Debug.Log($"[MujocoBridge] enlazados {found.Count}/{order.Count}:\n  " + string.Join("\n  ", found));
        if (missing.Count > 0)
            Debug.LogWarning("[MujocoBridge] sin pieza en el prefab (objetos de escena se dibujan como primitivas): " + string.Join(", ", missing));
    }

    Vector3 LocalPos(Transform t) => transform.InverseTransformPoint(t.position);
    Quaternion LocalRot(Transform t) => Quaternion.Inverse(transform.rotation) * t.rotation;

    // restJson = MuJoCo body poses at qpos0, which is the pose the prefab was built in
    void Calibrate(JObject restJson)
    {
        foreach (var name in order)
        {
            if (!links.TryGetValue(name, out Link L) || restJson[name] == null) continue;
            Read((JArray)restJson[name], out Vector3 pm, out Quaternion qm);
            L.restPos = LocalPos(L.t);
            L.restRot = LocalRot(L.t);
            Quaternion inv = Quaternion.Inverse(qm);
            L.offRot = inv * L.restRot;
            L.offPos = inv * (L.restPos - pm);
        }
        calibrated = true;
        Debug.Log("[MujocoBridge] calibrado contra la pose de reposo (qpos0) de MuJoCo");
    }

    [ContextMenu("Recalibrar")]
    void Recalibrate()
    {
        foreach (var L in links.Values)
        {
            if (!calibrated) break;
            L.t.position = transform.TransformPoint(L.restPos);
            L.t.rotation = transform.rotation * L.restRot;
        }
        calibrated = false;
    }

    static void Read(JArray a, out Vector3 p, out Quaternion q)
    {
        p = new Vector3((float)a[0], (float)a[1], (float)a[2]);
        q = new Quaternion((float)a[3], (float)a[4], (float)a[5], (float)a[6]);
    }

    void ApplyState()
    {
        var pkt = Interlocked.Exchange(ref latest, null);
        if (pkt == null) return;
        lastPacketTime = Time.time;
        JObject st = pkt.st;

        simTime = (float)st["t_sim"];
        masterOk = (bool)st["master_ok"];
        clutch = (bool)st["clutch"];
        pitchMargin = (float)st["pitch_margin"];
        jawForce = (float)st["jaw_contact_force"];
        contacts = (int)st["ncon"];
        var tq = (JArray)st["tool_q"];
        for (int i = 0; i < toolQ.Length && i < tq.Count; i++) toolQ[i] = (float)tq[i];
        if (st["master"] is JObject m)
        {
            var sing = (JObject)m["sing"];
            wristSing = (float)sing["wrist"];
            elbowSing = (float)sing["elbow"];
            shoulderSing = (float)sing["shoulder"];
            if (masterView != null) masterView.SetJointAngles(m["q"].ToObject<float[]>());
        }
        if (st["scene"] is JArray scene) UpdatePrims(scene);

        var bodiesJson = (JObject)st["bodies"];
        if (!bound) Bind(bodiesJson);
        if (!calibrated)
        {
            if (!(st["rest"] is JObject rest)) return;   // arrives every ~0.5 s
            Calibrate(rest);
        }

        foreach (var name in order)
        {
            if (!links.TryGetValue(name, out Link L)) continue;
            Read((JArray)bodiesJson[name], out Vector3 pm, out Quaternion qm);
            Vector3 p = pm + qm * L.offPos;
            Quaternion q = qm * L.offRot;
            L.t.position = transform.TransformPoint(p);
            L.t.rotation = transform.rotation * q;
        }
        int shown = UpdateContacts((JArray)st["contacts"]);
        UpdateHighlight();

        double tApply = Now();
        lastSeq = pkt.seq;
        holdMs = (float)((tApply - pkt.tRecv) * 1e3);
        latencyMs = (float)((tApply - pkt.tSend) * 1e3);   // valid only with Python on this machine
        if (logging)
            frameLog.WriteLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1},{2:F6},{3:F6},{4:F6},{5:F3},{6},{7},{8},{9:F6},{10:F6},{11:F6},{12:F6}",
                Time.frameCount, pkt.seq, pkt.tSend, pkt.tRecv, tApply, Time.unscaledDeltaTime * 1e3f,
                stale, lost, shown, toolQ[0], toolQ[1], toolQ[2], toolQ[3]));
    }

    // contact = [body1, body2, x, y, z, fn, fx, fy, fz], already in Unity axes
    int UpdateContacts(JArray list)
    {
        int n = showContacts && list != null ? list.Count : 0;
        while (markers.Count < n) markers.Add(NewPrimitive(PrimitiveType.Sphere, "contact").GetComponent<Renderer>());
        for (int i = 0; i < markers.Count; i++)
        {
            var r = markers[i];
            r.gameObject.SetActive(i < n);
            if (i >= n) continue;
            var c = (JArray)list[i];
            string b1 = (string)c[0], b2 = (string)c[1];
            var p = new Vector3((float)c[2], (float)c[3], (float)c[4]);
            float fn = (float)c[5];
            var f = new Vector3((float)c[6], (float)c[7], (float)c[8]);
            bool j1 = b1.Contains("gripper"), j2 = b2.Contains("gripper");
            Color col = j1 && j2 ? jawJawColor : j1 || j2 ? jawObjectColor : otherContactColor;

            r.transform.localPosition = p;
            r.transform.localScale = Vector3.one * markerSize * (1f + Mathf.Clamp01(Mathf.Abs(fn) / forceForMaxSize));
            SetColor(r, col);
            Debug.DrawRay(transform.TransformPoint(p), transform.TransformDirection(f) * (markerSize * 2f / forceForMaxSize), col);
        }
        return n;
    }

    // scene = primitives MuJoCo has but the prefab doesn't (e.g. tissue)
    void UpdatePrims(JArray scene)
    {
        foreach (JObject g in scene)
        {
            string name = (string)g["name"];
            bool box = (string)g["type"] == "box";
            if (!prims.TryGetValue(name, out Transform t))
            {
                t = NewPrimitive(box ? PrimitiveType.Cube : PrimitiveType.Sphere, name).transform;
                prims[name] = t;
                var c = (JArray)g["rgba"];
                SetColor(t.GetComponent<Renderer>(), new Color((float)c[0], (float)c[1], (float)c[2], (float)c[3]));
            }
            Read((JArray)g["pose"], out Vector3 p, out Quaternion q);
            var s = (JArray)g["size"];
            t.localPosition = p;
            t.localRotation = q;
            t.localScale = box ? 2f * new Vector3((float)s[0], (float)s[1], (float)s[2]) : Vector3.one * 2f * (float)s[0];
        }
    }

    // visual only: Unity must not run its own physics on these
    GameObject NewPrimitive(PrimitiveType type, string name)
    {
        var go = GameObject.CreatePrimitive(type);
        go.name = "mj_" + name;
        Destroy(go.GetComponent<Collider>());
        go.transform.SetParent(transform, false);
        return go;
    }

    void SetColor(Renderer r, Color c)
    {
        r.GetPropertyBlock(mpb);
        mpb.SetColor("_BaseColor", c);
        mpb.SetColor("_Color", c);
        r.SetPropertyBlock(mpb);
    }

    void UpdateHighlight()
    {
        if (highlightRenderers == null) return;
        Color c;
        if (clutch || !masterOk) c = clutchColor;
        else
        {
            float health = Mathf.Min(Mathf.Min(wristSing, elbowSing, shoulderSing),
                                     Mathf.Min(pitchMargin, 1f - Mathf.Clamp01(jawForce / jawForceLimitN)));
            c = health < badThreshold ? badColor : health < warnThreshold ? warnColor : okColor;
        }
        foreach (var r in highlightRenderers)
            if (r != null) SetColor(r, c);
    }

    void OnGUI()
    {
        string estado = !connected ? "MuJoCo SIN SEÑAL (¿corre el .py?)" : !masterOk ? "UR3e SIN SEÑAL (esclavo congelado)"
                      : clutch ? "CLUTCH (reposiciona el UR3e)" : "Siguiendo al UR3e";
        GUI.Label(new Rect(10, 10, 700, 140),
            $"{estado}   t={simTime:F2}s  piezas={links.Count}  contactos={contacts}\n" +
            $"UR3e — muñeca {wristSing:F2} · codo {elbowSing:F2} · hombro {shoulderSing:F2}   (0 = singular)\n" +
            $"Herramienta — roll {toolQ[0] * Mathf.Rad2Deg:F1}° · pitch {toolQ[1] * Mathf.Rad2Deg:F1}° · " +
            $"dedos {toolQ[2] * Mathf.Rad2Deg:F1}°/{toolQ[3] * Mathf.Rad2Deg:F1}° · F mandíbula {jawForce:F2} N\n" +
            $"Bridge — seq {lastSeq} · perdidos {lost} · no mostrados {stale} · desorden {outOfOrder} · " +
            $"latencia {latencyMs:F2} ms (espera en Unity {holdMs:F2} ms)\n" +
            "Espacio: clutch · E/D: abrir/cerrar pinza · R: reset");
    }

    void OnDestroy()
    {
        rx?.Close();
        tx?.Close();
        if (logging)
        {
            while (recvRows.TryDequeue(out string row)) recvLog.WriteLine(row);
            recvLog.Close();
            frameLog.Close();
        }
    }
}

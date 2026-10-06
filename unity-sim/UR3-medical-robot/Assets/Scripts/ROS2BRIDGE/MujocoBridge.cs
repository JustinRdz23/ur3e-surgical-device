// MujocoBridge.cs — Unity (lado visual) · v4
// La pinza simulada es ESCLAVA del UR3e real. Unity no calcula movimiento:
// solo manda BOTONES y aplica las poses que calcula MuJoCo.
//
//   Espacio (mantener) = clutch   ·   E / D (mantener) = abrir / cerrar pinza   ·   R = reset
//
// v4: ya no confía en cómo esté armado el prefab. Coloca cada pieza EXACTAMENTE donde
// la pone MuJoCo, y corrige cada malla automáticamente (sirve con high_res o low_res:
// la malla low_res de tool_roll_link viene recorrida 21.6 cm respecto a la high_res).
//
// Uso:
//  1. Script en la raíz de la herramienta (DvrkTool). Apaga los componentes DvrkJoint.
//  2. Clic derecho en el componente → "Alinear con MuJoCo (q=0)"  → arma la herramienta sin Play.
//     Clic derecho → "Apuntar Main Camera a la pinza"           → encuadra la punta.
//  3. Play y luego: python3 mujoco_unity_bridge.py --demo   (o --arm justi)

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

    [Header("Mallas")]
    [Tooltip("El importador OBJ de Unity invierte X. Si la pinza se ve espejeada, desmárcalo.")]
    public bool objImporterFlipsX = true;

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
    public bool connected, masterOk, clutch;
    public float simTime, wristSing = 1f, elbowSing = 1f, shoulderSing = 1f, pitchMargin = 1f, jawForce;
    public int contacts;
    public long lastSeq, received, lost, stale, outOfOrder;
    public float latencyMs, holdMs;
    public float[] toolQ = new float[4];

    // ---------- datos del modelo (calculados desde hello_unity.xml) ----------
    // pose q=0 de cada body, ya en marco Unity (relativa a la raíz), y centro del bbox
    // de la malla high_res en coordenadas OBJ (lo que MuJoCo espera).
    struct Q0
    {
        public Vector3 pos; public Quaternion rot; public Vector3 objCenter;
        public Q0(Vector3 p, Quaternion r, Vector3 c) { pos = p; rot = r; objCenter = c; }
    }
    static readonly string[] Order = { "tool_main_insert", "tool_roll_link", "tool_pitch_link",
                                       "tool_yaw_link", "gripper_left", "gripper_right" };
    static readonly Dictionary<string, Q0> Rest = new Dictionary<string, Q0> {
        {"tool_main_insert", new Q0(new Vector3(0.000000f,0.000000f,0.000000f), new Quaternion(-0.7071068f,0f,0f,0.7071068f), new Vector3(0.003693f,-0.000827f,0.000000f))},
        {"tool_roll_link",   new Q0(new Vector3(0.465891f,0.000000f,0.000000f), new Quaternion(-0.5f,-0.5f,-0.5f,0.5f),       new Vector3(0.000000f,0.000000f,-0.237554f))},
        {"tool_pitch_link",  new Q0(new Vector3(0.465891f,0.000000f,0.000000f), new Quaternion(0f,0f,0f,1f),                   new Vector3(0.004156f,0.000000f,-0.000076f))},
        {"tool_yaw_link",    new Q0(new Vector3(0.474883f,0.000000f,-0.000013f), new Quaternion(-0.7071068f,0f,0f,0.7071068f), new Vector3(0.000001f,0.000000f,0.000121f))},
        {"gripper_left",     new Q0(new Vector3(0.474883f,0.000000f,-0.000013f), new Quaternion(-0.5f,0.5f,-0.5f,0.5f),        new Vector3(0.000001f,0.003620f,0.000469f))},
        {"gripper_right",    new Q0(new Vector3(0.474883f,0.000000f,-0.000013f), new Quaternion(-0.5f,0.5f,-0.5f,0.5f),        new Vector3(-0.000001f,0.003620f,-0.000495f))},
    };
    static readonly Vector3 TipLocal = new Vector3(0.4775f, 0f, 0f);   // punta de las mandíbulas

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
        public bool meshOnNode;                 // la malla está en el mismo objeto que el link
        public Vector3 nodeOffPos; public Quaternion nodeOffRot = Quaternion.identity;
    }
    readonly Dictionary<string, Link> links = new Dictionary<string, Link>();
    readonly List<Renderer> markers = new List<Renderer>();
    readonly Dictionary<string, Transform> prims = new Dictionary<string, Transform>();
    bool bound;

    // same clock as Python's time.perf_counter() on Windows (QueryPerformanceCounter)
    static double Now() => Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;

    // ================= enlazar y corregir mallas =================
    static string ShortName(string n)
    {
        int i = n.IndexOf("_Shape");
        return i > 0 ? n.Substring(0, i) : n;
    }

    // rotación que lleva vértices importados por Unity al marco del body MuJoCo (en Unity)
    Quaternion MeshRot => new Quaternion(0f, 0.70710678f, 0.70710678f, 0f);
    Vector3 MeshScale => objImporterFlipsX ? Vector3.one : new Vector3(-1f, 1f, 1f);
    static Vector3 MjToUnity(Vector3 v) => new Vector3(v.x, v.z, v.y);

    Vector3 ObjDelta(MeshFilter mf, string key)
    {
        if (mf == null || mf.sharedMesh == null) return Vector3.zero;
        Vector3 b = mf.sharedMesh.bounds.center;
        Vector3 actual = objImporterFlipsX ? new Vector3(-b.x, b.y, b.z) : b;
        Vector3 d = Rest[key].objCenter - actual;
        return d.magnitude < 0.002f ? Vector3.zero : d;   // <2 mm: misma malla que MuJoCo
    }

    void Bind()
    {
        links.Clear();
        var all = GetComponentsInChildren<Transform>(true);
        var nodes = new HashSet<Transform>();
        var log = new List<string>(); var missing = new List<string>();

        foreach (var key in Order)
        {
            Transform hit = null;
            foreach (var t in all)   // preferir el objeto que NO es solo malla (el "link")
                if (t != transform && ShortName(t.name) == key && t.childCount > 0) { hit = t; break; }
            if (hit == null)
                foreach (var t in all)
                    if (t != transform && ShortName(t.name) == key) { hit = t; break; }
            if (hit == null) { missing.Add(key); continue; }
            links[key] = new Link { t = hit };
            nodes.Add(hit);
        }

        foreach (var kv in links)
        {
            string key = kv.Key; Link L = kv.Value;
            var mfSelf = L.t.GetComponent<MeshFilter>();
            if (mfSelf != null)
            {   // malla en el propio link: el offset va en la pose
                L.meshOnNode = true;
                L.nodeOffRot = MeshRot;
                L.nodeOffPos = MjToUnity(ObjDelta(mfSelf, key));
                L.t.localScale = MeshScale;
                log.Add($"{key} → {L.t.name} (malla en el nodo){Shift(L.nodeOffPos)}");
                continue;
            }
            int nMesh = 0; Vector3 shift = Vector3.zero;
            foreach (Transform c in L.t)
            {
                if (nodes.Contains(c)) continue;
                var mf = c.GetComponentInChildren<MeshFilter>(true);
                if (mf == null) continue;
                // si la malla está más profunda, corregimos el hijo directo que la contiene
                shift = MjToUnity(ObjDelta(mf, key));
                c.localPosition = shift;
                c.localRotation = MeshRot;
                c.localScale = MeshScale;
                if (mf.transform != c) { mf.transform.localPosition = Vector3.zero; mf.transform.localRotation = Quaternion.identity; mf.transform.localScale = Vector3.one; }
                nMesh++;
            }
            log.Add($"{key} → {L.t.name} ({nMesh} malla){Shift(shift)}");
        }
        bound = true;
        Debug.Log($"[MujocoBridge] enlazados {links.Count}/{Order.Length}:\n  " + string.Join("\n  ", log));
        if (missing.Count > 0)
            Debug.LogWarning("[MujocoBridge] no encontré en el prefab: " + string.Join(", ", missing));
    }

    static string Shift(Vector3 s) =>
        s == Vector3.zero ? "" : $"  · malla recorrida {s.magnitude * 100f:F1} cm (¿low_res?) → corregida";

    void SetBodyPose(string key, Vector3 p, Quaternion q)
    {
        if (!links.TryGetValue(key, out Link L)) return;
        Vector3 pos = p; Quaternion rot = q;
        if (L.meshOnNode) { pos = p + q * L.nodeOffPos; rot = q * L.nodeOffRot; }
        L.t.position = transform.TransformPoint(pos);
        L.t.rotation = transform.rotation * rot;
    }

    [ContextMenu("Alinear con MuJoCo (q=0)")]
    public void AlignRest()
    {
#if UNITY_EDITOR
        UnityEditor.Undo.RegisterFullObjectHierarchyUndo(gameObject, "Alinear con MuJoCo");
#endif
        Bind();
        foreach (var key in Order) SetBodyPose(key, Rest[key].pos, Rest[key].rot);
    }

    [ContextMenu("Apuntar Main Camera a la pinza")]
    public void AimCamera()
    {
        var cam = Camera.main;
        if (cam == null) { Debug.LogWarning("[MujocoBridge] no hay Main Camera (tag MainCamera)"); return; }
#if UNITY_EDITOR
        UnityEditor.Undo.RecordObjects(new Object[] { cam.transform, cam }, "Apuntar cámara");
#endif
        Vector3 tip = transform.TransformPoint(TipLocal);
        cam.transform.position = tip + transform.TransformVector(new Vector3(0.05f, 0.035f, -0.06f));
        cam.transform.LookAt(tip, transform.up);
        cam.nearClipPlane = 0.002f;   // con 0.3 m (default) la pinza queda recortada
        cam.fieldOfView = 40f;
    }

    // ================= runtime =================
    void Start()
    {
        mpb = new MaterialPropertyBlock();
        AlignRest();
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

        var bodies = (JObject)st["bodies"];
        foreach (var key in Order)                       // padre -> hijo
            foreach (var kv in bodies)
                if (ShortName(kv.Key) == key)
                {
                    var a = (JArray)kv.Value;            // [px,py,pz,qx,qy,qz,qw] marco Unity
                    SetBodyPose(key, new Vector3((float)a[0], (float)a[1], (float)a[2]),
                                     new Quaternion((float)a[3], (float)a[4], (float)a[5], (float)a[6]));
                    break;
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
        var kb = Keyboard.current;
        string teclas = kb == null ? "SIN TECLADO (Input System)"
            : $"clutch {(kb[clutchKey].isPressed ? "■" : "□")} · abrir {(kb[jawOpenKey].isPressed ? "■" : "□")} · cerrar {(kb[jawCloseKey].isPressed ? "■" : "□")}";
        GUI.Label(new Rect(10, 10, 700, 160),
            $"{estado}   t={simTime:F2}s  piezas={links.Count}  contactos={contacts}\n" +
            $"Teclas que ve Unity — {teclas}\n" +
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

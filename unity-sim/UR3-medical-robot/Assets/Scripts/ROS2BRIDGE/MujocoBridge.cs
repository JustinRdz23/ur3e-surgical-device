// MujocoBridge.cs — Unity (lado visual) · v3 (auto-calibración sobre prefab existente)
// La pinza simulada es ESCLAVA del UR3e real. Unity no calcula movimiento:
// solo manda BOTONES y aplica las poses que calcula MuJoCo.
//
//   Espacio (mantener) = clutch   ·   E / D (mantener) = abrir / cerrar pinza   ·   R = reset
//
// Setup:
//  1. Agrega este script al GameObject raíz de la herramienta (p.ej. "DvrkTool").
//  2. Las piezas se buscan por nombre EN CUALQUIER NIVEL de su jerarquía:
//     vale el nombre completo del <body> MuJoCo o solo la parte antes de "_Shape"
//     (tool_main_insert, tool_roll_link, tool_pitch_link, tool_yaw_link,
//      gripper_left, gripper_right). La consola dice cuáles encontró.
//  3. Auto-calibración: la pose que tenga el prefab al dar Play se toma como q = 0
//     de MuJoCo. No importa cómo estén orientadas/escaladas tus mallas.
//     (Clic derecho en el componente → "Recalibrar" si hace falta.)
//  4. Package Manager: com.unity.nuget.newtonsoft-json.
//  5. Play + `python3 mujoco_unity_bridge.py --model hello_unity.xml --demo`
//     (o --arm justi con el UR3e / URSim).

using System.Collections.Generic;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;
using UnityEngine;

public class MujocoBridge : MonoBehaviour
{
    [Header("Red")]
    public int statePort = 5005;   // MuJoCo -> Unity
    public int cmdPort = 5006;     // Unity -> MuJoCo (solo botones)
    public string pythonHost = "127.0.0.1";

    [Header("Botones")]
    public KeyCode clutchKey = KeyCode.Space;
    public KeyCode jawOpenKey = KeyCode.E;
    public KeyCode jawCloseKey = KeyCode.D;
    public KeyCode resetKey = KeyCode.R;

    [Header("Indicador visual (opcional)")]
    public Renderer[] highlightRenderers;          // p.ej. mandíbulas
    public Color okColor = new Color(0.2f, 0.8f, 0.3f);
    public Color warnColor = new Color(1f, 0.75f, 0.1f);
    public Color badColor = new Color(0.9f, 0.15f, 0.15f);
    public Color clutchColor = new Color(0.4f, 0.6f, 1f);
    [Range(0f, 1f)] public float warnThreshold = 0.35f;
    [Range(0f, 1f)] public float badThreshold = 0.10f;
    public float jawForceLimitN = 5f;

    [Header("Estado (solo lectura)")]
    public bool connected, masterOk, clutch, calibrated;
    public float simTime, wristSing = 1f, elbowSing = 1f, shoulderSing = 1f, pitchMargin = 1f, jawForce;
    public int contacts;

    UdpClient rx, tx;
    Thread rxThread;
    volatile string latest;
    float lastPacketTime = -10f;
    MaterialPropertyBlock mpb;

    class Link
    {
        public Transform t;
        public Vector3 restPos;   public Quaternion restRot;   // pose Unity en q=0 (marco del root)
        public Vector3 offPos;    public Quaternion offRot;    // offset MuJoCo->Unity
    }
    readonly Dictionary<string, Link> links = new Dictionary<string, Link>();   // clave = nombre body MuJoCo
    readonly List<string> order = new List<string>();                          // orden padre->hijo de MuJoCo
    bool bound;

    void Start()
    {
        mpb = new MaterialPropertyBlock();
        tx = new UdpClient();
        rx = new UdpClient(statePort);
        rxThread = new Thread(ReceiveLoop) { IsBackground = true };
        rxThread.Start();
    }

    void ReceiveLoop()
    {
        var ep = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try { latest = Encoding.UTF8.GetString(rx.Receive(ref ep)); }
            catch (SocketException) { return; }
            catch (System.ObjectDisposedException) { return; }
        }
    }

    void Update()
    {
        SendButtons();
        ApplyState();
        connected = Time.time - lastPacketTime < 0.5f;
    }

    void SendButtons()
    {
        bool reset = Input.GetKeyDown(resetKey);
        string json = string.Format(CultureInfo.InvariantCulture,
            "{{\"buttons\":{{\"clutch\":{0},\"jaw_open\":{1},\"jaw_close\":{2}}}{3}}}",
            B(Input.GetKey(clutchKey)), B(Input.GetKey(jawOpenKey)), B(Input.GetKey(jawCloseKey)),
            reset ? ",\"reset\":true" : "");
        byte[] b = Encoding.UTF8.GetBytes(json);
        tx.Send(b, b.Length, pythonHost, cmdPort);
    }

    static string B(bool v) => v ? "true" : "false";

    // --- enlazar bodies MuJoCo con transforms del prefab por nombre ---
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
            Debug.LogWarning("[MujocoBridge] sin pieza en el prefab (no se moverán): " + string.Join(", ", missing));
    }

    // pose relativa al root
    Vector3 LocalPos(Transform t) => transform.InverseTransformPoint(t.position);
    Quaternion LocalRot(Transform t) => Quaternion.Inverse(transform.rotation) * t.rotation;

    void Calibrate(JObject bodiesJson)
    {
        foreach (var name in order)
        {
            if (!links.TryGetValue(name, out Link L)) continue;
            Read((JArray)bodiesJson[name], out Vector3 pm, out Quaternion qm);
            L.restPos = LocalPos(L.t);
            L.restRot = LocalRot(L.t);
            Quaternion inv = Quaternion.Inverse(qm);
            L.offRot = inv * L.restRot;
            L.offPos = inv * (L.restPos - pm);
        }
        calibrated = true;
        Debug.Log("[MujocoBridge] calibrado: la pose actual del prefab = q0 de MuJoCo");
    }

    [ContextMenu("Recalibrar")]
    void Recalibrate()
    {
        foreach (var L in links.Values)
        {   // regresar a la pose de reposo y volver a calibrar con el siguiente paquete
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
        string s = latest;
        if (s == null) return;
        latest = null;
        lastPacketTime = Time.time;

        JObject st = JObject.Parse(s);
        simTime = (float)st["t"];
        masterOk = (bool)st["master_ok"];
        clutch = (bool)st["clutch"];
        pitchMargin = (float)st["pitch_margin"];
        jawForce = (float)st["jaw_contact_force"];
        contacts = (int)st["ncon"];
        if (st["master"] is JObject m)
        {
            var sing = (JObject)m["sing"];
            wristSing = (float)sing["wrist"];
            elbowSing = (float)sing["elbow"];
            shoulderSing = (float)sing["shoulder"];
        }

        var bodiesJson = (JObject)st["bodies"];
        if (!bound) Bind(bodiesJson);
        if (!calibrated) Calibrate(bodiesJson);

        // padre -> hijo, en poses de mundo: pose_unity = pose_mujoco * offset
        foreach (var name in order)
        {
            if (!links.TryGetValue(name, out Link L)) continue;
            Read((JArray)bodiesJson[name], out Vector3 pm, out Quaternion qm);
            Vector3 p = pm + qm * L.offPos;
            Quaternion q = qm * L.offRot;
            L.t.position = transform.TransformPoint(p);
            L.t.rotation = transform.rotation * q;
        }
        UpdateHighlight();
    }

    // Azul = clutch / sin robot. Si no: peor de (singularidades del UR3e maestro,
    // margen de pitch de la herramienta, fuerza en mandíbulas).
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
        {
            if (r == null) continue;
            r.GetPropertyBlock(mpb);
            mpb.SetColor("_BaseColor", c); // URP Lit
            mpb.SetColor("_Color", c);     // Built-in
            r.SetPropertyBlock(mpb);
        }
    }

    void OnGUI()
    {
        string estado = !connected ? "MuJoCo SIN SEÑAL (¿corre el .py?)" : !masterOk ? "UR3e SIN SEÑAL (esclavo congelado)"
                      : clutch ? "CLUTCH (reposiciona el UR3e)" : "Siguiendo al UR3e";
        GUI.Label(new Rect(10, 10, 600, 110),
            $"{estado}   t={simTime:F2}s  piezas={links.Count}  contactos={contacts}\n" +
            $"UR3e — muñeca {wristSing:F2} · codo {elbowSing:F2} · hombro {shoulderSing:F2}   (0 = singular)\n" +
            $"Herramienta — margen pitch {pitchMargin:F2} · F mandíbula {jawForce:F2} N\n" +
            "Espacio: clutch · E/D: abrir/cerrar pinza · R: reset");
    }

    void OnDestroy()
    {
        rx?.Close();
        tx?.Close();
    }
}

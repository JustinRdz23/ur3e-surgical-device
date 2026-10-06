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

    [Header("Mallas")]
    [Tooltip("El importador OBJ de Unity invierte X. Si la pinza se ve espejeada, desmárcalo.")]
    public bool objImporterFlipsX = true;

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
    public bool connected, masterOk, clutch;
    public float simTime, wristSing = 1f, elbowSing = 1f, shoulderSing = 1f, pitchMargin = 1f, jawForce;
    public int contacts;

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

    class Link
    {
        public Transform t;
        public bool meshOnNode;                 // la malla está en el mismo objeto que el link
        public Vector3 nodeOffPos; public Quaternion nodeOffRot = Quaternion.identity;
    }
    readonly Dictionary<string, Link> links = new Dictionary<string, Link>();
    bool bound;

    UdpClient rx, tx;
    Thread rxThread;
    volatile string latest;
    float lastPacketTime = -10f;
    MaterialPropertyBlock mpb;

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
        string json = string.Format(CultureInfo.InvariantCulture,
            "{{\"buttons\":{{\"clutch\":{0},\"jaw_open\":{1},\"jaw_close\":{2}}}{3}}}",
            B(Input.GetKey(clutchKey)), B(Input.GetKey(jawOpenKey)), B(Input.GetKey(jawCloseKey)),
            Input.GetKeyDown(resetKey) ? ",\"reset\":true" : "");
        byte[] b = Encoding.UTF8.GetBytes(json);
        tx.Send(b, b.Length, pythonHost, cmdPort);
    }

    static string B(bool v) => v ? "true" : "false";

    void ApplyState()
    {
        string s = latest;
        if (s == null) return;
        latest = null;
        lastPacketTime = Time.time;
        if (!bound) Bind();

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

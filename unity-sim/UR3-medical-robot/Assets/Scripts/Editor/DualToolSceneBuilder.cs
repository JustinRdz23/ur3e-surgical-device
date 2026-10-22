// Builds Assets/Scenes/DualToolScene.unity from the layout that dual_tool_gen.py
// exports (MuJoCo "home" keyframe, already in Unity axes): the Operation board and
// pieces, both dVRK tools and the endoscope camera. MuJoCo stays the source of
// truth; re-run this menu after re-running dual_tool_gen.py.
//
// Look: the studio lights of ProductShot (key/fill/rim, baked reflections, neutral
// tonemapping) and its tool materials; silver board, pieces in a blue-to-white
// ramp, inside the tissue cavity that dual_tool_gen.py builds, lit also by a lamp
// on the endoscope. No depth of field: an endoscope view must stay sharp.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public static class DualToolSceneBuilder
{
    const string MeshDir = "Assets/Models/Operation";
    const string LayoutPath = MeshDir + "/dual_tool_layout.json";
    const string ToolPrefabPath = "Assets/Prefabs/DvrkTool.prefab";
    const string ScenePath = "Assets/Scenes/DualToolScene.unity";
    const string Dir = "Assets/DualTool";
    const string ProductShotMaterials = "Assets/ProductShot/Materials";

    const string BoardBody = "operation_board_body";
    const string CavityBody = "cavity_body";
    static readonly Color BackdropColor = new Color(0.86f, 0.87f, 0.89f);   // probe background
    static readonly Color CavityDark = new Color(0.16f, 0.05f, 0.05f);       // camera clear colour
    static readonly Color Tissue = new Color(0.78f, 0.36f, 0.38f);
    static readonly Color BoardSilver = new Color(0.80f, 0.82f, 0.85f);
    static readonly Color PieceBlue = new Color(0.05f, 0.24f, 0.72f);
    static readonly Color PieceWhite = new Color(0.95f, 0.97f, 1.00f);

    // softboxes only show up in reflections: the camera culls this layer, the probe doesn't
    const int SoftboxLayer = 31;

    // Rotation that takes vertices as imported by Unity into the MuJoCo body frame
    // (same as MujocoBridge.MeshRot / DvrkToolBuilder).
    static readonly Quaternion MeshRot = new Quaternion(0f, 0.70710678f, 0.70710678f, 0f);

    // Keys per tool: left hand drives the left tool, right hand the right one. R resets the sim.
    struct ToolKeys { public Key open, close, clutch; public Vector2 hud; }
    static readonly Dictionary<string, ToolKeys> Keys = new Dictionary<string, ToolKeys> {
        {"left",  new ToolKeys { open = Key.E, close = Key.D, clutch = Key.LeftShift,  hud = new Vector2(10f, 10f) }},
        {"right", new ToolKeys { open = Key.I, close = Key.K, clutch = Key.RightShift, hud = new Vector2(10f, 190f) }},
    };

    // tool link -> ProductShot material (white housing, dark shaft, steel wrist and jaws)
    static readonly Dictionary<string, string> ToolMaterials = new Dictionary<string, string> {
        {"tool_main_insert_Shape_IndexedFaceSet.030", "Housing"},
        {"tool_roll_link_Shape_IndexedFaceSet", "Shaft"},
        {"tool_pitch_link_Shape_IndexedFaceSet.014", "Wrist"},
        {"tool_yaw_link_Shape_IndexedFaceSet.015", "Wrist"},
        {"gripper_left_Shape_IndexedFaceSet.006", "Wrist"},
        {"gripper_right_Shape_IndexedFaceSet.022", "Wrist"},
    };

    [MenuItem("Tools/DVRK/Build Dual Tool Scene")]
    public static void Build()
    {
        if (!File.Exists(LayoutPath))
        {
            Debug.LogError($"[DualToolScene] {LayoutPath} not found: run dual_tool_gen.py first");
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        Directory.CreateDirectory(Dir + "/Materials");
        var layout = JObject.Parse(File.ReadAllText(LayoutPath));
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        BuildCamera((JObject)layout["camera"]);
        BuildLights();
        BuildSoftboxes();
        BuildVolume();
        Vector3 meshScale = ImporterFlipsX() ? Vector3.one : new Vector3(-1f, 1f, 1f);

        // MuJoCo's world origin. The bridges place every body relative to it.
        var world = new GameObject("MuJoCoWorld").transform;
        var bodiesRoot = new GameObject("MuJoCoBodies").transform;
        bodiesRoot.SetParent(world, false);
        var bodies = (JObject)layout["bodies"];
        var pieceNames = bodies.Properties().Select(p => p.Name)
                               .Where(n => n != BoardBody && n != CavityBody).ToList();
        var sceneProps = new List<GameObject>();   // everything the reflection bake must not see
        foreach (var kv in bodies)
        {
            var node = PlaceBody(kv.Key, (JArray)kv.Value, bodiesRoot, meshScale);
            if (node == null) continue;
            if (kv.Key == CavityBody)
            {
                // seen from inside only; casting shadows it would black out the studio lights
                Paint(node, TissueMaterial());
                foreach (var r in node.GetComponentsInChildren<Renderer>(true))
                    r.shadowCastingMode = ShadowCastingMode.Off;
                continue;
            }
            Paint(node, kv.Key == BoardBody ? BoardMaterial() : PieceMaterial(pieceNames.IndexOf(kv.Key), pieceNames.Count));
            sceneProps.Add(node.gameObject);
        }

        var toolPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(ToolPrefabPath);
        var tools = new List<GameObject>();
        bool first = true;
        foreach (var kv in (JObject)layout["tools"])
        {
            // The first tool also moves the board/pieces and draws contacts; the
            // second would only duplicate that work.
            tools.Add(PlaceTool(kv.Key, (JArray)kv.Value, toolPrefab, world, first ? bodiesRoot : null));
            first = false;
        }

        // the metal should reflect the pink cavity, not itself
        BakeReflections(sceneProps.Concat(tools).ToArray());

        Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
        EditorSceneManager.SaveScene(scene, ScenePath);
        Debug.Log($"[DualToolScene] built {ScenePath}: {bodies.Count} bodies, {tools.Count} tools");
    }

    static void ReadPose(JArray a, out Vector3 p, out Quaternion q)
    {
        p = new Vector3((float)a[0], (float)a[1], (float)a[2]);
        q = new Quaternion((float)a[3], (float)a[4], (float)a[5], (float)a[6]);
    }

    static Vector3 Vec(JToken t) => new Vector3((float)t[0], (float)t[1], (float)t[2]);

    // ---------------------------------------------------------------- bodies and tools

    // Board / piece: a node at the body pose, with the OBJ as a child rotated into the body frame.
    static Transform PlaceBody(string name, JArray pose, Transform parent, Vector3 meshScale)
    {
        var model = AssetDatabase.LoadAssetAtPath<GameObject>($"{MeshDir}/{name}.obj");
        if (model == null)
        {
            Debug.LogWarning($"[DualToolScene] no mesh for {name}");
            return null;
        }
        ReadPose(pose, out Vector3 p, out Quaternion q);
        var node = new GameObject(name).transform;
        node.SetParent(parent, false);
        node.SetPositionAndRotation(p, q);
        var mesh = ((GameObject)PrefabUtility.InstantiatePrefab(model, node)).transform;
        mesh.localPosition = Vector3.zero;
        mesh.localRotation = MeshRot;
        mesh.localScale = meshScale;
        return node;
    }

    // Tool: prefab root at the mount frame, links placed by MujocoBridge.AlignRest().
    // Both bridges share one UDP socket (MujocoBridge keeps it static) and each
    // reads only the bodies and state of its own prefix.
    static GameObject PlaceTool(string name, JArray pose, GameObject prefab, Transform world, Transform bodiesRoot)
    {
        ReadPose(pose, out Vector3 p, out Quaternion q);
        var tool = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        tool.name = $"DvrkTool_{name}";
        tool.transform.SetPositionAndRotation(p, q);
        // TryGetComponent, not "?? AddComponent": in the editor a missing component is a fake null
        if (!tool.TryGetComponent(out MujocoBridge bridge)) bridge = tool.AddComponent<MujocoBridge>();
        bridge.toolPrefix = name + "_";
        bridge.mujocoWorld = world;
        bridge.bodiesRoot = bodiesRoot;
        bridge.drawScene = bodiesRoot != null;
        if (Keys.TryGetValue(name, out ToolKeys k))
        {
            bridge.jawOpenKey = k.open;
            bridge.jawCloseKey = k.close;
            bridge.clutchKey = k.clutch;
            bridge.hudPosition = k.hud;
        }
        bridge.AlignRest();
        bridge.enabled = true;

        foreach (var t in tool.GetComponentsInChildren<Transform>(true))
            if (ToolMaterials.TryGetValue(t.name, out string matName))
                foreach (Transform c in t)   // the link's own meshes, not the links nested under it
                    if (!ToolMaterials.ContainsKey(c.name))
                        foreach (var r in c.GetComponentsInChildren<Renderer>(true))
                            r.sharedMaterial = ToolMaterial(matName);
        return tool;
    }

    static void Paint(Transform node, Material m)
    {
        foreach (var r in node.GetComponentsInChildren<Renderer>(true))
            r.sharedMaterials = Enumerable.Repeat(m, r.sharedMaterials.Length).ToArray();
    }

    // ---------------------------------------------------------------- materials

    static Material ToolMaterial(string name)
    {
        // reuse ProductShot's materials so both scenes look the same
        var m = AssetDatabase.LoadAssetAtPath<Material>($"{ProductShotMaterials}/{name}.mat");
        if (m != null) return m;
        return name switch
        {
            "Housing" => Mat("Housing", new Color(0.92f, 0.92f, 0.9f), 0f, 0.55f),
            "Shaft" => Mat("Shaft", new Color(0.09f, 0.09f, 0.1f), 0.85f, 0.62f),
            _ => Mat("Wrist", new Color(0.82f, 0.83f, 0.85f), 1f, 0.82f),
        };
    }

    static Material BoardMaterial() => Mat("Board_Silver", BoardSilver, 1f, 0.78f);

    // wet tissue: not metallic, quite glossy
    static Material TissueMaterial() => Mat("Cavity_Tissue", Tissue, 0f, 0.72f);

    // i-th of n pieces: deep blue for the first, near-white for the last
    static Material PieceMaterial(int i, int n)
    {
        float t = n > 1 ? i / (float)(n - 1) : 0f;
        return Mat($"Piece_{i + 1:D2}", Color.Lerp(PieceBlue, PieceWhite, t), 0f, 0.5f);
    }

    static Material Mat(string name, Color color, float metallic, float smoothness, bool unlit = false)
    {
        var path = $"{Dir}/Materials/{name}.mat";
        var shader = Shader.Find(unlit ? "Universal Render Pipeline/Unlit" : "Universal Render Pipeline/Lit");
        var m = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (m == null)
        {
            m = new Material(shader);
            AssetDatabase.CreateAsset(m, path);
        }
        m.shader = shader;
        m.SetColor("_BaseColor", color);
        if (!unlit)
        {
            m.SetFloat("_Metallic", metallic);
            m.SetFloat("_Smoothness", smoothness);
        }
        EditorUtility.SetDirty(m);
        return m;
    }

    // ---------------------------------------------------------------- studio (from ProductShot)

    static void BuildCamera(JObject cam)
    {
        var go = new GameObject("Endoscope Camera") { tag = "MainCamera" };
        var c = go.AddComponent<Camera>();
        go.AddComponent<AudioListener>();
        c.transform.SetPositionAndRotation(Vec(cam["pos"]), Quaternion.LookRotation(Vec(cam["forward"]), Vec(cam["up"])));
        c.fieldOfView = (float)cam["fovy"];   // both are vertical FOV
        c.nearClipPlane = 0.001f;             // the default 0.3 m would clip the whole scene
        c.farClipPlane = 10f;
        c.clearFlags = CameraClearFlags.SolidColor;
        c.backgroundColor = CavityDark;   // only visible through gaps in the cavity
        c.cullingMask = ~(1 << SoftboxLayer);
        if (!go.TryGetComponent(out UniversalAdditionalCameraData data)) data = go.AddComponent<UniversalAdditionalCameraData>();
        data.renderPostProcessing = true;
        data.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;
        data.antialiasingQuality = AntialiasingQuality.High;

        // endoscope lamp: rides with the camera, like the real scope's light guide
        var lamp = new GameObject("Endoscope Light").AddComponent<Light>();
        lamp.transform.SetParent(go.transform, false);
        lamp.type = LightType.Spot;
        lamp.spotAngle = 75f;
        lamp.innerSpotAngle = 40f;
        lamp.range = 1.5f;
        lamp.intensity = 1.2f;
        lamp.color = new Color(1f, 0.96f, 0.9f);
        lamp.shadows = LightShadows.Soft;
    }

    static void BuildLights()
    {
        // key from the front-left, the only one that casts shadows; fill low front-right; rim from behind for edges
        Light("Key", new Vector3(48, 35, 0), 2.2f, new Color(1f, 0.97f, 0.93f), true);
        Light("Fill", new Vector3(14, -55, 0), 0.55f, new Color(0.9f, 0.94f, 1f), false);
        Light("Rim", new Vector3(32, 180, 0), 1.6f, Color.white, false);

        RenderSettings.ambientMode = AmbientMode.Trilight;
        RenderSettings.ambientSkyColor = new Color(0.62f, 0.64f, 0.68f);
        RenderSettings.ambientEquatorColor = new Color(0.5f, 0.5f, 0.52f);
        RenderSettings.ambientGroundColor = new Color(0.25f, 0.25f, 0.26f);
        RenderSettings.skybox = null;
        DynamicGI.UpdateEnvironment();
    }

    static void Light(string name, Vector3 euler, float intensity, Color color, bool shadows)
    {
        var go = new GameObject(name);
        go.transform.rotation = Quaternion.Euler(euler);
        var l = go.AddComponent<Light>();
        l.type = LightType.Directional;
        l.intensity = intensity;
        l.color = color;
        l.shadows = shadows ? LightShadows.Soft : LightShadows.None;
        l.shadowStrength = 0.85f;
    }

    static void BuildSoftboxes()
    {
        var mat = AssetDatabase.LoadAssetAtPath<Material>($"{ProductShotMaterials}/Softbox.mat")
                  ?? Mat("Softbox", new Color(6f, 6f, 6f), 0f, 0f, unlit: true);
        Softbox("Softbox Top", new Vector3(0, 2.2f, -0.6f), new Vector3(2.4f, 1.2f, 1), mat);
        Softbox("Softbox Left", new Vector3(-2.2f, 1.0f, -0.8f), new Vector3(0.6f, 1.8f, 1), mat);
        Softbox("Softbox Right", new Vector3(2.2f, 0.9f, -0.5f), new Vector3(0.4f, 1.6f, 1), mat);
    }

    static void Softbox(string name, Vector3 pos, Vector3 scale, Material m)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name;
        go.layer = SoftboxLayer;
        Object.DestroyImmediate(go.GetComponent<Collider>());
        // a quad's visible side faces -Z: point +Z away from the scene
        go.transform.SetPositionAndRotation(pos, Quaternion.LookRotation(pos - new Vector3(0, 0.05f, 0)));
        go.transform.localScale = scale;
        go.GetComponent<MeshRenderer>().sharedMaterial = m;
    }

    static void BuildVolume()
    {
        var path = Dir + "/DualToolVolume.asset";
        AssetDatabase.DeleteAsset(path);
        var profile = ScriptableObject.CreateInstance<VolumeProfile>();
        AssetDatabase.CreateAsset(profile, path);

        var tm = profile.Add<Tonemapping>(true);
        tm.mode.Override(TonemappingMode.Neutral);
        var bloom = profile.Add<Bloom>(true);
        bloom.intensity.Override(0.15f);
        bloom.threshold.Override(1.1f);
        var vig = profile.Add<Vignette>(true);
        vig.intensity.Override(0.18f);
        vig.smoothness.Override(0.5f);
        foreach (var c in profile.components) AssetDatabase.AddObjectToAsset(c, profile);
        AssetDatabase.SaveAssets();

        var go = new GameObject("Post Processing");
        var v = go.AddComponent<Volume>();
        v.isGlobal = true;
        v.sharedProfile = profile;
    }

    // metal is black without something bright to reflect: capture backdrop + softboxes into a cubemap
    static void BakeReflections(GameObject[] hide)
    {
        var go = new GameObject("Reflection Probe");
        go.transform.position = new Vector3(0f, 0.15f, -0.1f);
        var probe = go.AddComponent<ReflectionProbe>();
        probe.size = new Vector3(10, 6, 10);
        probe.resolution = 512;
        probe.hdr = true;
        probe.cullingMask = ~0;
        probe.clearFlags = ReflectionProbeClearFlags.SolidColor;
        probe.backgroundColor = BackdropColor;

        var path = Dir + "/StudioReflection.exr";
        foreach (var h in hide) h.SetActive(false);
        try { Lightmapping.BakeReflectionProbe(probe, path); }
        finally { foreach (var h in hide) h.SetActive(true); }
        AssetDatabase.ImportAsset(path);
        var cube = AssetDatabase.LoadAssetAtPath<Cubemap>(path);
        probe.mode = ReflectionProbeMode.Custom;
        probe.customBakedTexture = cube;
        RenderSettings.defaultReflectionMode = DefaultReflectionMode.Custom;
        RenderSettings.customReflectionTexture = cube;
    }

    // Unity's OBJ importer has negated X in every version we tried; check instead of assuming.
    static bool ImporterFlipsX()
    {
        string path = $"{MeshDir}/operation_board_body.obj";
        float min = float.MaxValue;
        foreach (var line in File.ReadLines(path))
            if (line.StartsWith("v "))
                min = Mathf.Min(min, float.Parse(line.Split(' ')[1], System.Globalization.CultureInfo.InvariantCulture));
        var mesh = AssetDatabase.LoadAllAssetsAtPath(path).OfType<Mesh>().FirstOrDefault();
        return mesh == null || Mathf.Abs(mesh.bounds.max.x + min) < 1e-4f;
    }
}

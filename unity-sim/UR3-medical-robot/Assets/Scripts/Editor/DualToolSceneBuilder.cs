// Builds Assets/Scenes/DualToolScene.unity from the layout that dual_tool_gen.py
// exports (MuJoCo "home" keyframe, already in Unity axes): the Operation board and
// pieces, both dVRK tools and the endoscope camera. MuJoCo stays the source of
// truth; re-run this menu after re-running dual_tool_gen.py.

using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class DualToolSceneBuilder
{
    const string MeshDir = "Assets/Models/Operation";
    const string LayoutPath = MeshDir + "/dual_tool_layout.json";
    const string ToolPrefabPath = "Assets/Prefabs/DvrkTool.prefab";
    const string ScenePath = "Assets/Scenes/DualToolScene.unity";

    // Rotation that takes vertices as imported by Unity into the MuJoCo body frame
    // (same as MujocoBridge.MeshRot / DvrkToolBuilder).
    static readonly Quaternion MeshRot = new Quaternion(0f, 0.70710678f, 0.70710678f, 0f);

    [MenuItem("Tools/DVRK/Build Dual Tool Scene")]
    public static void Build()
    {
        if (!File.Exists(LayoutPath))
        {
            Debug.LogError($"[DualToolScene] {LayoutPath} not found: run dual_tool_gen.py first");
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

        var layout = JObject.Parse(File.ReadAllText(LayoutPath));
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        PlaceCamera((JObject)layout["camera"]);
        Vector3 meshScale = ImporterFlipsX() ? Vector3.one : new Vector3(-1f, 1f, 1f);

        var bodiesRoot = new GameObject("MuJoCoBodies").transform;
        foreach (var kv in (JObject)layout["bodies"])
            PlaceBody(kv.Key, (JArray)kv.Value, bodiesRoot, meshScale);

        var toolPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(ToolPrefabPath);
        foreach (var kv in (JObject)layout["tools"])
            PlaceTool(kv.Key, (JArray)kv.Value, toolPrefab);

        Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
        EditorSceneManager.SaveScene(scene, ScenePath);
        Debug.Log($"[DualToolScene] built {ScenePath}: {((JObject)layout["bodies"]).Count} bodies, " +
                  $"{((JObject)layout["tools"]).Count} tools");
    }

    static void ReadPose(JArray a, out Vector3 p, out Quaternion q)
    {
        p = new Vector3((float)a[0], (float)a[1], (float)a[2]);
        q = new Quaternion((float)a[3], (float)a[4], (float)a[5], (float)a[6]);
    }

    static Vector3 Vec(JToken t) => new Vector3((float)t[0], (float)t[1], (float)t[2]);

    static void PlaceCamera(JObject cam)
    {
        var c = Camera.main;
        c.name = "Endoscope Camera";
        c.transform.SetPositionAndRotation(Vec(cam["pos"]), Quaternion.LookRotation(Vec(cam["forward"]), Vec(cam["up"])));
        c.fieldOfView = (float)cam["fovy"];   // both are vertical FOV
        c.nearClipPlane = 0.001f;             // the default 0.3 m would clip the whole scene
        c.farClipPlane = 5f;
    }

    // Board / piece: a node at the body pose, with the OBJ as a child rotated into the body frame.
    static void PlaceBody(string name, JArray pose, Transform parent, Vector3 meshScale)
    {
        var model = AssetDatabase.LoadAssetAtPath<GameObject>($"{MeshDir}/{name}.obj");
        if (model == null)
        {
            Debug.LogWarning($"[DualToolScene] no mesh for {name}");
            return;
        }
        ReadPose(pose, out Vector3 p, out Quaternion q);
        var node = new GameObject(name).transform;
        node.SetParent(parent, false);
        node.SetPositionAndRotation(p, q);
        var mesh = ((GameObject)PrefabUtility.InstantiatePrefab(model, node)).transform;
        mesh.localPosition = Vector3.zero;
        mesh.localRotation = MeshRot;
        mesh.localScale = meshScale;
    }

    // Tool: prefab root at the mount frame, links placed by MujocoBridge.AlignRest().
    // The bridge is left disabled so the preview opens no UDP port; with two tools
    // both would try to bind 5005.
    static void PlaceTool(string name, JArray pose, GameObject prefab)
    {
        ReadPose(pose, out Vector3 p, out Quaternion q);
        var tool = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
        tool.name = $"DvrkTool_{name}";
        tool.transform.SetPositionAndRotation(p, q);
        var bridge = tool.GetComponent<MujocoBridge>() ?? tool.AddComponent<MujocoBridge>();
        bridge.AlignRest();
        bridge.enabled = false;
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

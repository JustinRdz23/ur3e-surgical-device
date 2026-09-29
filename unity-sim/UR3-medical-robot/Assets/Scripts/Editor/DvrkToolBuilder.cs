using System.Globalization;
using System.IO;
using System.Linq;
using System.Xml;
using UnityEditor;
using UnityEngine;

// Builds Assets/Prefabs/DvrkTool.prefab from Assets/Models/DVRK/dvrk_tool.xml (the MuJoCo MJCF),
// converting MuJoCo (right-handed, Z-up) to Unity (left-handed, Y-up) with x,y,z -> x,z,y.
public static class DvrkToolBuilder
{
    const string ModelDir = "Assets/Models/DVRK";
    const string MjcfPath = ModelDir + "/dvrk_tool.xml";
    const string PrefabPath = "Assets/Prefabs/DvrkTool.prefab";
    const string MaterialPath = ModelDir + "/DvrkSteel.mat";

    // Mesh child correction, set by DetectImporterFlipsX.
    static Quaternion meshRot;
    static Vector3 meshScale;

    [MenuItem("Tools/DVRK/Build Tool Prefab")]
    public static void Build()
    {
        var doc = new XmlDocument();
        doc.Load(MjcfPath);

        var meshFiles = doc.SelectNodes("//asset/mesh").Cast<XmlElement>()
            .ToDictionary(m => m.GetAttribute("name"), m => m.GetAttribute("file"));

        DetectImporterFlipsX(meshFiles.Values.First(f => f.Contains("pitch")));

        var material = LoadOrCreateMaterial();
        var root = new GameObject("DvrkTool");
        foreach (XmlElement body in doc.SelectNodes("/mujoco/worldbody/body"))
            BuildBody(body, root.transform, meshFiles, material);

        Directory.CreateDirectory(Path.GetDirectoryName(PrefabPath));
        PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
        Object.DestroyImmediate(root);
        Debug.Log("DVRK tool prefab written to " + PrefabPath);
    }

    static void BuildBody(XmlElement body, Transform parent, System.Collections.Generic.Dictionary<string, string> meshFiles, Material material)
    {
        var go = new GameObject(body.GetAttribute("name"));
        go.transform.SetParent(parent, false);
        go.transform.localPosition = ToUnity(ParseVec(body.GetAttribute("pos"), Vector3.zero));
        go.transform.localRotation = ToUnity(ParseQuat(body.GetAttribute("quat")));

        // A hinge rotates the body itself, so the joint sits on the same transform.
        var joint = body.ChildNodes.OfType<XmlElement>().FirstOrDefault(e => e.Name == "joint");
        if (joint != null && joint.GetAttribute("type") == "hinge")
        {
            var j = go.AddComponent<DvrkJoint>();
            j.unityAxis = ToUnity(ParseVec(joint.GetAttribute("axis"), Vector3.forward)).normalized;
            j.restLocalRotation = go.transform.localRotation;
            var range = joint.GetAttribute("range").Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
            if (range.Length == 2)
            {
                j.minRad = float.Parse(range[0], CultureInfo.InvariantCulture);
                j.maxRad = float.Parse(range[1], CultureInfo.InvariantCulture);
            }
            else { j.minRad = -Mathf.PI; j.maxRad = Mathf.PI; }
        }

        foreach (var geom in body.ChildNodes.OfType<XmlElement>().Where(e => e.Name == "geom" && e.GetAttribute("type") == "mesh"))
        {
            var meshPath = ModelDir + "/" + meshFiles[geom.GetAttribute("mesh")];
            var mesh = AssetDatabase.LoadAllAssetsAtPath(meshPath).OfType<Mesh>().First();
            var visual = new GameObject(geom.GetAttribute("name"));
            visual.transform.SetParent(go.transform, false);
            visual.transform.localRotation = meshRot;
            visual.transform.localScale = meshScale;
            visual.AddComponent<MeshFilter>().sharedMesh = mesh;
            visual.AddComponent<MeshRenderer>().sharedMaterial = material;
        }

        foreach (var child in body.ChildNodes.OfType<XmlElement>().Where(e => e.Name == "body"))
            BuildBody(child, go.transform, meshFiles, material);
    }

    // Unity's OBJ importer may or may not negate X on import. Compare the imported bounds with the
    // raw vertices of an asymmetric mesh to find out, then pick the child transform that lands the
    // mesh on x,z,y of the file's coordinates either way.
    static void DetectImporterFlipsX(string objFile)
    {
        var path = ModelDir + "/" + objFile;
        float min = float.MaxValue, max = float.MinValue;
        foreach (var line in File.ReadLines(path))
        {
            if (!line.StartsWith("v ")) continue;
            var x = float.Parse(line.Split(' ')[1], CultureInfo.InvariantCulture);
            if (x < min) min = x;
            if (x > max) max = x;
        }
        var b = AssetDatabase.LoadAllAssetsAtPath(path).OfType<Mesh>().First().bounds;
        bool flipped = Mathf.Abs(b.min.x + max) < 1e-4f && Mathf.Abs(b.max.x + min) < 1e-4f;
        bool same = Mathf.Abs(b.min.x - min) < 1e-4f && Mathf.Abs(b.max.x - max) < 1e-4f;
        if (!flipped && !same) Debug.LogWarning("Could not tell whether the OBJ importer flips X; assuming it does.");

        if (flipped || !same)
        {
            // imported (-x,y,z) -> wanted (x,z,y): 180 degrees about (0,1,1)
            meshRot = Quaternion.AngleAxis(180f, new Vector3(0, 1, 1).normalized);
            meshScale = Vector3.one;
        }
        else
        {
            // imported (x,y,z) -> wanted (x,z,y): mirror Z, then 90 degrees about X
            meshRot = Quaternion.AngleAxis(90f, Vector3.right);
            meshScale = new Vector3(1, 1, -1);
        }
        Debug.Log("OBJ importer flips X: " + (flipped || !same));
    }

    static Material LoadOrCreateMaterial()
    {
        var m = AssetDatabase.LoadAssetAtPath<Material>(MaterialPath);
        if (m != null) return m;
        m = new Material(Shader.Find("Universal Render Pipeline/Lit")) { color = new Color(0.75f, 0.77f, 0.8f) };
        AssetDatabase.CreateAsset(m, MaterialPath);
        return m;
    }

    static Vector3 ParseVec(string s, Vector3 fallback)
    {
        if (string.IsNullOrEmpty(s)) return fallback;
        var p = s.Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries)
            .Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        return new Vector3(p[0], p[1], p[2]);
    }

    // MJCF quats are w x y z.
    static Quaternion ParseQuat(string s)
    {
        if (string.IsNullOrEmpty(s)) return Quaternion.identity;
        var p = s.Split(new[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries)
            .Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();
        return new Quaternion(p[1], p[2], p[3], p[0]);
    }

    static Vector3 ToUnity(Vector3 v) => new Vector3(v.x, v.z, v.y);
    static Quaternion ToUnity(Quaternion q) => new Quaternion(-q.x, -q.z, -q.y, q.w);
}

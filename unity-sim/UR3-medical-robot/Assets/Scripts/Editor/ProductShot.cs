// Product renders of the DVRK tool: builds a studio scene (Assets/Scenes/ProductShot.unity) from code and
// renders a fixed shot list to Renders/ProductShot/*.png (outside Assets, so the PNGs are not imported).
// The scene is rebuilt on every render: tweak this file, not the scene.
//
//   Tools/Product Shot/Render Preview (1080p)  ·  Render Final (4K)
//   Or write "preview" / "final" into Temp/ProductShot.request while ProductShot is the open scene;
//   the result goes to Temp/ProductShot.status.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class ProductShot
{
    const string ScenePath = "Assets/Scenes/ProductShot.unity";
    const string Dir = "Assets/ProductShot";
    const string PrefabPath = "Assets/Prefabs/DvrkTool.prefab";
    const string SourceRpAsset = "Assets/Settings/PC_RPAsset.asset";
    const string RequestFile = "Temp/ProductShot.request";
    const string StatusFile = "Temp/ProductShot.status";
    const string OutDir = "Renders/ProductShot";

    // softboxes only show up in reflections: the render camera culls this layer, the reflection probe doesn't
    const int SoftboxLayer = 31;

    const string Housing = "tool_main_insert_Shape_IndexedFaceSet.030";
    const string Shaft = "tool_roll_link_Shape_IndexedFaceSet";
    const string Pitch = "tool_pitch_link_Shape_IndexedFaceSet.014";
    const string Yaw = "tool_yaw_link_Shape_IndexedFaceSet.015";
    const string JawL = "gripper_left_Shape_IndexedFaceSet.006";
    const string JawR = "gripper_right_Shape_IndexedFaceSet.022";

    // spin of the whole tool about its shaft, to pick which face of the housing rests on the floor
    const float ToolRollDeg = 0f;

    enum Target { All, Housing, Wrist }

    class Shot
    {
        public string name;
        public Target target;
        public float azimuth, elevation, fov, margin;   // azimuth 0 = camera in front (-Z), + = toward the wrist end
        public float roll, pitch, jaw;                 // tool joints [rad]
        public float explode;                          // wrist parts pulled apart along the shaft [m]
        public float aperture;                         // f-stop for depth of field; 0 = everything sharp
        public float focalMm = 85f;
    }

    static readonly Shot[] Shots =
    {
        new Shot { name = "01_hero", target = Target.All, azimuth = -32, elevation = 20, fov = 24, margin = 0.08f,
                   pitch = 0.35f, jaw = 0.4f },
        new Shot { name = "02_wrist_macro", target = Target.Wrist, azimuth = 38, elevation = 14, fov = 20, margin = 0.22f,
                   pitch = 0.6f, jaw = 0.55f, aperture = 4f, focalMm = 100f },
        new Shot { name = "03_housing", target = Target.Housing, azimuth = -55, elevation = 28, fov = 28, margin = 0.16f,
                   pitch = 0.35f, jaw = 0.4f, aperture = 5.6f, focalMm = 70f },
        new Shot { name = "04_wrist_exploded", target = Target.Wrist, azimuth = 0, elevation = 8, fov = 20, margin = 0.12f,
                   jaw = 0.3f, explode = 0.008f },
    };

    class Rig
    {
        public Camera cam;
        public Transform tool;
        public Dictionary<string, Transform> bodies;
        public Dictionary<string, Vector3> restLocal;
        public VolumeProfile profile;
        public UniversalRenderPipelineAsset rp;
    }

    static double nextPoll;

    static ProductShot() => EditorApplication.update += Poll;

    [MenuItem("Tools/Product Shot/Render Preview (1080p)")]
    static void RenderPreview() => RenderFromMenu(1920, 1080);

    [MenuItem("Tools/Product Shot/Render Final (4K)")]
    static void RenderFinal() => RenderFromMenu(3840, 2160);

    static void RenderFromMenu(int w, int h)
    {
        if (SceneManager.GetActiveScene().path != ScenePath && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            return;
        var files = Render(w, h);
        EditorUtility.RevealInFinder(files[0]);
    }

    // Unity -batchmode -projectPath . -executeMethod ProductShot.Batch [-final]
    public static void Batch()
    {
        int code = 0;
        try
        {
            bool final = Environment.GetCommandLineArgs().Contains("-final");
            Render(final ? 3840 : 1920, final ? 2160 : 1080);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            code = 1;
        }
        EditorApplication.Exit(code);
    }

    static void Poll()
    {
        if (EditorApplication.timeSinceStartup < nextPoll) return;
        nextPoll = EditorApplication.timeSinceStartup + 1.0;
        if (!File.Exists(RequestFile) || EditorApplication.isCompiling || EditorApplication.isPlayingOrWillChangePlaymode)
            return;

        var mode = File.ReadAllText(RequestFile).Trim();
        File.Delete(RequestFile);
        try
        {
            // never switch scenes behind the user's back: they may have unsaved work open
            if (SceneManager.GetActiveScene().path != ScenePath)
                throw new InvalidOperationException("open " + ScenePath + " first (or use the Tools/Product Shot menu once)");
            var files = mode == "final" ? Render(3840, 2160) : Render(1920, 1080);
            File.WriteAllText(StatusFile, "ok\n" + string.Join("\n", files) + "\n");
        }
        catch (Exception e)
        {
            File.WriteAllText(StatusFile, "error\n" + e + "\n");
            Debug.LogException(e);
        }
    }

    static List<string> Render(int width, int height)
    {
        var rig = Build();
        Directory.CreateDirectory(OutDir);
        var prevQuality = QualitySettings.renderPipeline;
        var files = new List<string>();
        try
        {
            QualitySettings.renderPipeline = rig.rp;
            foreach (var shot in Shots)
            {
                var path = Path.GetFullPath(Path.Combine(OutDir, shot.name + ".png"));
                RenderShot(rig, shot, width, height, path);
                files.Add(path);
            }
        }
        finally
        {
            QualitySettings.renderPipeline = prevQuality;
            Pose(rig, new Shot());
        }
        EditorSceneManager.SaveScene(SceneManager.GetActiveScene(), ScenePath);
        Debug.Log($"Product shots ({width}x{height}) written to {Path.GetFullPath(OutDir)}");
        return files;
    }

    static void RenderShot(Rig rig, Shot shot, int width, int height, string path)
    {
        Pose(rig, shot);
        var b = TargetBounds(rig, shot.target);

        var cam = rig.cam;
        cam.fieldOfView = shot.fov;
        cam.aspect = (float)width / height;
        float az = shot.azimuth * Mathf.Deg2Rad, el = shot.elevation * Mathf.Deg2Rad;
        var dir = new Vector3(Mathf.Cos(el) * Mathf.Sin(az), Mathf.Sin(el), -Mathf.Cos(el) * Mathf.Cos(az));
        float dist = FitDistance(cam, b, dir, shot.margin);
        cam.transform.position = b.center + dir * dist;
        cam.transform.LookAt(b.center);
        cam.nearClipPlane = Mathf.Max(0.002f, dist * 0.02f);

        rig.rp.shadowDistance = dist + 1.0f;
        var dof = rig.profile.components.OfType<DepthOfField>().First();
        dof.active = shot.aperture > 0;
        dof.focusDistance.Override(dist);
        dof.aperture.Override(Mathf.Max(1f, shot.aperture));
        dof.focalLength.Override(shot.focalMm);

        // supersample 2x, then a bilinear half-size blit is an exact 2x2 box filter
        const int ss = 2;
        var big = new RenderTexture(width * ss, height * ss, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        var small = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        var prevActive = RenderTexture.active;
        try
        {
            cam.targetTexture = big;
            cam.Render();
            Graphics.Blit(big, small);
            RenderTexture.active = small;
            var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
        }
        finally
        {
            cam.targetTexture = null;
            RenderTexture.active = prevActive;
            big.Release();
            small.Release();
        }
    }

    // closest camera distance that keeps every corner of b inside the frame minus the margin
    static float FitDistance(Camera cam, Bounds b, Vector3 dir, float margin)
    {
        var corners = new List<Vector3>();
        for (int i = 0; i < 8; i++)
            corners.Add(b.center + Vector3.Scale(b.extents, new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1)));
        float lo = 0.001f, hi = 50f;
        for (int it = 0; it < 40; it++)
        {
            float d = 0.5f * (lo + hi);
            cam.transform.position = b.center + dir * d;
            cam.transform.LookAt(b.center);
            bool fits = corners.All(c =>
            {
                var v = cam.WorldToViewportPoint(c);
                return v.z > 0 && v.x >= margin && v.x <= 1 - margin && v.y >= margin && v.y <= 1 - margin;
            });
            if (fits) hi = d; else lo = d;
        }
        return hi;
    }

    static Bounds TargetBounds(Rig rig, Target t)
    {
        IEnumerable<Renderer> rs = t switch
        {
            Target.Housing => OwnRenderers(rig.bodies[Housing]),
            Target.Wrist => rig.bodies[Pitch].GetComponentsInChildren<Renderer>(),
            _ => rig.tool.GetComponentsInChildren<Renderer>(),
        };
        var list = rs.ToList();
        var b = list[0].bounds;
        foreach (var r in list.Skip(1)) b.Encapsulate(r.bounds);
        return b;
    }

    static void Pose(Rig rig, Shot s)
    {
        SetJoint(rig, Shaft, s.roll);
        SetJoint(rig, Pitch, s.pitch);
        SetJoint(rig, JawL, s.jaw);
        SetJoint(rig, JawR, s.jaw);

        // explode from the rest layout; parent first, so each child's world offset is absolute
        foreach (var n in ExplodeChain) rig.bodies[n].localPosition = rig.restLocal[n];
        if (s.explode == 0) return;
        var axis = ShaftAxis(rig);
        var world = ExplodeChain.ToDictionary(n => n, n => rig.bodies[n].position);
        float[] steps = { 1, 2, 3, 3 };
        for (int i = 0; i < ExplodeChain.Length; i++)
            rig.bodies[ExplodeChain[i]].position = world[ExplodeChain[i]] + axis * (s.explode * steps[i]);
    }

    static readonly string[] ExplodeChain = { Pitch, Yaw, JawL, JawR };

    static void SetJoint(Rig rig, string body, float rad)
    {
        var j = rig.bodies[body].GetComponent<DvrkJoint>();
        if (j != null) j.AngleRad = rad;
    }

    static Vector3 ShaftAxis(Rig rig) =>
        (rig.bodies[Yaw].position - TargetBoundsCenter(OwnRenderers(rig.bodies[Housing]))).normalized;

    static Vector3 TargetBoundsCenter(IEnumerable<Renderer> rs)
    {
        var l = rs.ToList();
        var b = l[0].bounds;
        foreach (var r in l.Skip(1)) b.Encapsulate(r.bounds);
        return b.center;
    }

    // a body's own meshes, without the bodies nested under it
    static IEnumerable<Renderer> OwnRenderers(Transform body) =>
        body.Cast<Transform>().Select(c => c.GetComponent<Renderer>()).Where(r => r != null);

    // ---------------------------------------------------------------- scene

    static Rig Build()
    {
        Directory.CreateDirectory(Dir);
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var rig = new Rig { rp = MakeRenderPipelineAsset() };
        PlaceTool(rig);
        BuildBackdrop();
        BuildLights();
        BuildSoftboxes();
        rig.profile = BuildVolume();
        rig.cam = BuildCamera();
        BakeReflections(rig.tool.gameObject);

        EditorSceneManager.SaveScene(scene, ScenePath);
        return rig;
    }

    static void PlaceTool(Rig rig)
    {
        var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath);
        var tool = rig.tool = ((GameObject)PrefabUtility.InstantiatePrefab(prefab)).transform;
        var bodies = rig.bodies = tool.GetComponentsInChildren<Transform>().ToDictionary(t => t.name, t => t);
        rig.restLocal = ExplodeChain.ToDictionary(n => n, n => bodies[n].localPosition);
        // the saved prefab has identity rest rotations on its joints (old DvrkToolBuilder bug): repair the instance
        foreach (var kv in DvrkToolBuilder.RestRotations())
        {
            if (!bodies.TryGetValue(kv.Key, out var t)) continue;
            if (t.TryGetComponent(out DvrkJoint j)) j.restLocalRotation = kv.Value;
            t.localRotation = kv.Value;
        }
        Pose(rig, new Shot());

        // shaft along +X (housing left, wrist right), then spin about the shaft
        tool.rotation = Quaternion.AngleAxis(ToolRollDeg, Vector3.right) * Quaternion.FromToRotation(ShaftAxis(rig), Vector3.right);

        var all = TargetBounds(rig, Target.All);
        tool.position += new Vector3(-all.center.x, -all.min.y, -all.center.z);

        var housing = Mat("Housing", new Color(0.92f, 0.92f, 0.9f), 0f, 0.55f);
        var shaft = Mat("Shaft", new Color(0.09f, 0.09f, 0.1f), 0.85f, 0.62f);
        var wrist = Mat("Wrist", new Color(0.82f, 0.83f, 0.85f), 1f, 0.82f);
        Assign(bodies[Housing], housing);
        Assign(bodies[Shaft], shaft);
        Assign(bodies[Pitch], wrist);
        Assign(bodies[Yaw], wrist);
        Assign(bodies[JawL], wrist);
        Assign(bodies[JawR], wrist);
    }

    static void Assign(Transform body, Material m)
    {
        foreach (var r in OwnRenderers(body)) r.sharedMaterial = m;
    }

    // cyclorama: floor that curves into a back wall, so there is no horizon line
    static void BuildBackdrop()
    {
        const float width = 8f, front = -4f, coveZ = 0.6f, radius = 0.9f, height = 4f;
        var profile = new List<Vector2> { new Vector2(front, 0) };      // (z, y)
        for (int i = 0; i <= 24; i++)
        {
            float a = i / 24f * Mathf.PI / 2;
            profile.Add(new Vector2(coveZ + Mathf.Sin(a) * radius, radius - Mathf.Cos(a) * radius));
        }
        profile.Add(new Vector2(coveZ + radius, height));

        var verts = new List<Vector3>();
        var tris = new List<int>();
        foreach (var p in profile)
        {
            verts.Add(new Vector3(-width / 2, p.y, p.x));
            verts.Add(new Vector3(width / 2, p.y, p.x));
        }
        for (int i = 0; i < profile.Count - 1; i++)
        {
            int l0 = 2 * i, r0 = l0 + 1, l1 = l0 + 2, r1 = l0 + 3;
            tris.AddRange(new[] { l0, l1, r0, l1, r1, r0 });
        }
        var mesh = new Mesh { name = "Cyclorama", vertices = verts.ToArray(), triangles = tris.ToArray() };
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        var meshPath = Dir + "/Cyclorama.asset";
        AssetDatabase.DeleteAsset(meshPath);
        AssetDatabase.CreateAsset(mesh, meshPath);

        var go = new GameObject("Backdrop");
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        go.AddComponent<MeshRenderer>().sharedMaterial = Mat("Backdrop", new Color(0.86f, 0.87f, 0.89f), 0f, 0.22f);
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
        var mat = Mat("Softbox", new Color(6f, 6f, 6f), 0f, 0f, unlit: true);
        Softbox("Softbox Top", new Vector3(0, 2.2f, -0.6f), new Vector3(2.4f, 1.2f, 1), mat);
        Softbox("Softbox Left", new Vector3(-2.2f, 1.0f, -0.8f), new Vector3(0.6f, 1.8f, 1), mat);
        Softbox("Softbox Right", new Vector3(2.2f, 0.9f, -0.5f), new Vector3(0.4f, 1.6f, 1), mat);
    }

    static void Softbox(string name, Vector3 pos, Vector3 scale, Material m)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
        go.name = name;
        go.layer = SoftboxLayer;
        UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
        // a quad's visible side faces -Z: point +Z away from the tool
        go.transform.SetPositionAndRotation(pos, Quaternion.LookRotation(pos - new Vector3(0, 0.05f, 0)));
        go.transform.localScale = scale;
        go.GetComponent<MeshRenderer>().sharedMaterial = m;
    }

    static VolumeProfile BuildVolume()
    {
        var path = Dir + "/ProductShotVolume.asset";
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
        var dof = profile.Add<DepthOfField>(true);
        dof.mode.Override(DepthOfFieldMode.Bokeh);
        foreach (var c in profile.components) AssetDatabase.AddObjectToAsset(c, profile);
        AssetDatabase.SaveAssets();

        var go = new GameObject("Post Processing");
        var v = go.AddComponent<Volume>();
        v.isGlobal = true;
        v.sharedProfile = profile;
        return profile;
    }

    static Camera BuildCamera()
    {
        var go = new GameObject("Product Camera");
        var cam = go.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = new Color(0.86f, 0.87f, 0.89f);
        cam.cullingMask = ~(1 << SoftboxLayer);
        cam.farClipPlane = 30f;
        if (!go.TryGetComponent(out UniversalAdditionalCameraData data))
            data = go.AddComponent<UniversalAdditionalCameraData>();
        data.renderPostProcessing = true;
        data.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;
        data.antialiasingQuality = AntialiasingQuality.High;
        go.tag = "MainCamera";
        return cam;
    }

    // metal is black without something bright to reflect: capture backdrop + softboxes into a cubemap
    static void BakeReflections(GameObject tool)
    {
        var go = new GameObject("Reflection Probe");
        go.transform.position = new Vector3(0, 0.3f, -0.2f);
        var probe = go.AddComponent<ReflectionProbe>();
        probe.size = new Vector3(10, 6, 10);
        probe.resolution = 512;
        probe.hdr = true;
        probe.cullingMask = ~0;
        probe.clearFlags = ReflectionProbeClearFlags.SolidColor;
        probe.backgroundColor = new Color(0.86f, 0.87f, 0.89f);

        var path = Dir + "/StudioReflection.exr";
        tool.SetActive(false);
        try { Lightmapping.BakeReflectionProbe(probe, path); }
        finally { tool.SetActive(true); }
        AssetDatabase.ImportAsset(path);
        var cube = AssetDatabase.LoadAssetAtPath<Cubemap>(path);
        probe.mode = ReflectionProbeMode.Custom;
        probe.customBakedTexture = cube;
        RenderSettings.defaultReflectionMode = DefaultReflectionMode.Custom;
        RenderSettings.customReflectionTexture = cube;
    }

    // a copy of the project's URP asset with shadows tuned for a 0.5 m object; only active while rendering
    static UniversalRenderPipelineAsset MakeRenderPipelineAsset()
    {
        var path = Dir + "/ProductShot_RPAsset.asset";
        AssetDatabase.DeleteAsset(path);
        AssetDatabase.CopyAsset(SourceRpAsset, path);
        var asset = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(path);
        var so = new SerializedObject(asset);
        so.FindProperty("m_MainLightShadowmapResolution").intValue = 8192;
        so.FindProperty("m_ShadowCascadeCount").intValue = 1;
        so.FindProperty("m_SoftShadowsSupported").boolValue = true;
        so.FindProperty("m_SoftShadowQuality").intValue = 3;
        so.FindProperty("m_SupportsHDR").boolValue = true;
        so.FindProperty("m_RenderScale").floatValue = 1f;
        so.ApplyModifiedPropertiesWithoutUndo();
        AssetDatabase.SaveAssets();
        return asset;
    }

    static Material Mat(string name, Color color, float metallic, float smoothness, bool unlit = false)
    {
        Directory.CreateDirectory(Dir + "/Materials");
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
}

using UnityEngine;

// Passive view of the UR3e master: MujocoBridge feeds it the real joint angles.
public class UR3Controller : MonoBehaviour
{
    public Transform[] joints = new Transform[6];

    // local rotation axis per joint; flip sign to match the imported model
    public Vector3[] axes =
    {
        Vector3.up, Vector3.right, Vector3.right, Vector3.right, Vector3.up, Vector3.right
    };

    private readonly float[] jointAngles = new float[6];
    private readonly Quaternion[] restRotations = new Quaternion[6];

    void Start()
    {
        for (int i = 0; i < 6; i++)
            if (joints[i] != null) restRotations[i] = joints[i].localRotation;
    }

    void Update()
    {
        for (int i = 0; i < 6; i++)
        {
            if (joints[i] == null) continue;
            joints[i].localRotation = restRotations[i] * Quaternion.AngleAxis(jointAngles[i] * Mathf.Rad2Deg, axes[i]);
        }
    }

    public void SetJointAngles(float[] angles)
    {
        if (angles != null && angles.Length == 6)
            System.Array.Copy(angles, jointAngles, 6);
    }

    public float[] GetJointAngles()
    {
        return jointAngles;
    }
}

using UnityEngine;

public class UR3Controller : MonoBehaviour
{
    public Transform[] joints = new Transform[6];

    private float[] jointAngles = new float[6];

    private float[] minAngles = { -3.14f, -3.14f, -3.14f, -3.14f, -3.14f, -3.14f };
    private float[] maxAngles = { 3.14f, 3.14f, 3.14f, 3.14f, 3.14f, 3.14f };

    public float rotationSpeed = 0.05f;

    void Start()
    {
        Debug.Log("UR3Controller initialized with " + joints.Length + " joints");
    }

    void Update()
    {
        if (Input.GetKey(KeyCode.W)) jointAngles[0] += rotationSpeed;
        if (Input.GetKey(KeyCode.S)) jointAngles[0] -= rotationSpeed;

        if (Input.GetKey(KeyCode.A)) jointAngles[1] += rotationSpeed;
        if (Input.GetKey(KeyCode.D)) jointAngles[1] -= rotationSpeed;

        if (Input.GetKey(KeyCode.Q)) jointAngles[2] += rotationSpeed;
        if (Input.GetKey(KeyCode.E)) jointAngles[2] -= rotationSpeed;

        ApplyJointLimits();
        UpdateJointTransforms();
    }

    void ApplyJointLimits()
    {
        for (int i = 0; i < 6; i++)
        {
            jointAngles[i] = Mathf.Clamp(jointAngles[i], minAngles[i], maxAngles[i]);
        }
    }

    void UpdateJointTransforms()
    {
        for (int i = 0; i < 6; i++)
        {
            if (joints[i] != null)
            {
                joints[i].localRotation = Quaternion.Euler(
                    jointAngles[i] * Mathf.Rad2Deg,
                    0,
                    0
                );
            }
        }
    }

    public void SetJointAngles(float[] angles)
    {
        if (angles.Length == 6)
        {
            System.Array.Copy(angles, jointAngles, 6);
        }
    }

    public float[] GetJointAngles()
    {
        return jointAngles;
    }
}

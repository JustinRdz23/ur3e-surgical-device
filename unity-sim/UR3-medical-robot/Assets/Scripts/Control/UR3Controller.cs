using UnityEngine;

public class UR3Controller : MonoBehaviour
{
    // Articulaciones del UR3 (6 DOF)
    public Transform[] joints = new Transform[6];
    
    // Ángulos actuales (radianes)
    private float[] jointAngles = new float[6];
    
    // Límites de articulaciones UR3 (en radianes)
    private float[] minAngles = { -3.14f, -3.14f, -3.14f, -3.14f, -3.14f, -3.14f };
    private float[] maxAngles = { 3.14f, 3.14f, 3.14f, 3.14f, 3.14f, 3.14f };
    
    // Velocidad de rotación (para testing)
    public float rotationSpeed = 0.05f;
    
    void Start()
    {
        Debug.Log("UR3Controller initialized with " + joints.Length + " joints");
    }
    
    void Update()
    {
        // Control temporal con teclado (para testing)
        // W/S = Joint 0 (base rotation)
        if (Input.GetKey(KeyCode.W)) jointAngles[0] += rotationSpeed;
        if (Input.GetKey(KeyCode.S)) jointAngles[0] -= rotationSpeed;
        
        // A/D = Joint 1 (shoulder)
        if (Input.GetKey(KeyCode.A)) jointAngles[1] += rotationSpeed;
        if (Input.GetKey(KeyCode.D)) jointAngles[1] -= rotationSpeed;
        
        // Q/E = Joint 2 (elbow)
        if (Input.GetKey(KeyCode.Q)) jointAngles[2] += rotationSpeed;
        if (Input.GetKey(KeyCode.E)) jointAngles[2] -= rotationSpeed;
        
        // Aplicar límites y actualizar articulaciones
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
                // Convierte radianes a grados y aplica rotación
                joints[i].localRotation = Quaternion.Euler(
                    jointAngles[i] * Mathf.Rad2Deg, 
                    0, 
                    0
                );
            }
        }
    }
    
    // Método que ROS2 llamará después
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

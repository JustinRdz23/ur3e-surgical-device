using UnityEngine;

public class DvrkJoint : MonoBehaviour
{
    public Vector3 unityAxis = Vector3.up;
    public float minRad;
    public float maxRad;
    public Quaternion restLocalRotation = Quaternion.identity;

    [SerializeField] float angleRad;

    public float AngleRad
    {
        get => angleRad;
        set
        {
            angleRad = Mathf.Clamp(value, minRad, maxRad);
            transform.localRotation = restLocalRotation * Quaternion.AngleAxis(-angleRad * Mathf.Rad2Deg, unityAxis);
        }
    }

    // MujocoBridge owns the pose when present; OnValidate runs even with the component disabled
    void OnValidate()
    {
        if (GetComponentInParent<MujocoBridge>(true) != null) return;
        AngleRad = angleRad;
    }
}

using UnityEngine;

// Revolute joint of the dVRK tool, built from an MJCF <joint type="hinge">.
// MuJoCo is right-handed Z-up, Unity is left-handed Y-up (mapping used: x,y,z -> x,z,y).
// A rotation of +theta about a MuJoCo axis becomes -theta about the swapped axis in Unity.
public class DvrkJoint : MonoBehaviour
{
    public Vector3 unityAxis = Vector3.up;   // MuJoCo axis, already converted to Unity
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

    void OnValidate() => AngleRad = angleRad;
}

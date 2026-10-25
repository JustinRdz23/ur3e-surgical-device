using UnityEngine;

public class HMDCalibration : MonoBehaviour
{
    [Header("References")]
    public Transform trackedCamera;
    public Transform calibrationTarget;

    [Header("Controls")]
    public KeyCode calibrationKey = KeyCode.C;

    private Vector3 initialPosition;
    private Quaternion initialRotation;

    private void Awake()
    {
        initialPosition = transform.position;
        initialRotation = transform.rotation;
    }

    private void Update()
    {
        if (Input.GetKeyDown(calibrationKey))
        {
            Calibrate();
        }
    }

    [ContextMenu("Calibrate HMD")]
    public void Calibrate()
    {
        if (trackedCamera == null || calibrationTarget == null)
        {
            Debug.LogError(
                "[HMDCalibration] Tracked Camera and Calibration Target must be assigned."
            );
            return;
        }

        Quaternion rotationCorrection =
            calibrationTarget.rotation *
            Quaternion.Inverse(trackedCamera.rotation);

        transform.rotation =
            rotationCorrection * transform.rotation;

        Vector3 positionCorrection =
            calibrationTarget.position -
            trackedCamera.position;

        transform.position += positionCorrection;

        Debug.Log(
            $"[HMDCalibration] Calibration complete.\n" +
            $"Camera position: {trackedCamera.position}\n" +
            $"Target position: {calibrationTarget.position}"
        );
    }

    [ContextMenu("Reset Calibration")]
    public void ResetCalibration()
    {
        transform.position = initialPosition;
        transform.rotation = initialRotation;

        Debug.Log("[HMDCalibration] Calibration reset.");
    }
}

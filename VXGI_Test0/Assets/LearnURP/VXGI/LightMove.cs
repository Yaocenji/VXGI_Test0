using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LightMove : MonoBehaviour
{
    [Header("一圈时间")]
    public float oneCycleTime = 20;

    [Header("变化的轴")]
    [Range(0, 2)]
    public int axis = 0;
    
    [Header("其他轴")]
    public float otherAxisValue = 45;

    [Header("基础光强")] 
    public float baseLightStrength = 15.0f;
    [Header("光强波动范围")] 
    public float lightStrengthAmplitude = 10.0f;

    private Light _light;
    
    private Transform tr;
    void Start()
    {
        tr = transform;
        _light = GetComponent<Light>();
    }

    // Update is called once per frame
    void Update()
    {
        if (axis == 0)
            tr.rotation = Quaternion.Euler(Time.time * (360.0f / oneCycleTime) % 360, otherAxisValue, otherAxisValue);
        if (axis == 1)
            tr.rotation = Quaternion.Euler(otherAxisValue, Time.time * (360.0f / oneCycleTime) % 360, otherAxisValue);
        if (axis == 2)
            tr.rotation = Quaternion.Euler(otherAxisValue, otherAxisValue, Time.time * (360.0f / oneCycleTime) % 360);

        _light.intensity = baseLightStrength + Mathf.Sin(Time.time) * lightStrengthAmplitude;
    }
}

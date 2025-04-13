using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UI;

[ExecuteAlways]
public class VoxelGI : MonoBehaviour
{
    // 单例
    public static VoxelGI instance;
    // 体素范围：以摄像机为中心的正方体
    // 设置范围大小
    [Header("体素分辨率")]
    public int voxTexSize = 256;
    [Header("体素大小")] 
    public float voxSize = 0.125f;
    [Header("LOD级数")]
    [Range(1, 8)]
    public int lodLevels = 5;

    private float voxAreaSize => voxTexSize * voxSize;

    public int voxNumber {
        get
        {
            int ans = 0;
            int curTexSize = voxTexSize;
            for (int i = 0; i < lodLevels; i++)
            {
                ans += curTexSize * curTexSize * curTexSize;
                curTexSize /= 2;
            }
            return ans;
        }
    }
    // 初始化3d体素纹理的计算着色器
    public ComputeShader manageVoxelDataCS;
    // 设置三维纹理的大小
    // 摄像机位置
    private Transform camTrans;

    private Vector3 LookUpPos;
    private Vector3 LookForwardPos;
    private Vector3 LookRightPos;
    // 三个摄像机的transform
    public Transform LookUpTrans;
    public Transform LookForwardTrans;
    public Transform LookRightTrans;
    // 三个方向的view矩阵
    private Matrix4x4 LookUp;
    private Matrix4x4 LookForward;
    private Matrix4x4 LookRight;
    // 正交变换阵
    private Matrix4x4 OrthoTrans;
    // 三个变量的int索引
    private int upMatIdx, forwardMatIdx, rightMatIdx;
    private int camPosIdx;
    // 一张三维纹理（表面是三维纹理，实际上是一维的计算buffer）
    public ComputeBuffer voxelBuffer;
    // 体素零点
    private Vector3 zeroPos;
    // 体素中心
    private Vector3 centerPos => zeroPos + voxAreaSize / 2.0f * Vector3.one;
    
    void Start()
    {
        // 单例引用
        instance = this;
        // 初始化体素网格信息
        int initKernel = manageVoxelDataCS.FindKernel("Clear");
        Debug.Log(voxNumber);
        voxelBuffer = new ComputeBuffer(voxNumber, 24);
        Graphics.SetRandomWriteTarget(1, voxelBuffer, false);
        manageVoxelDataCS.SetBuffer(initKernel, "VoxelTexture", voxelBuffer);
        manageVoxelDataCS.SetInt("voxTexSize", voxTexSize);
        manageVoxelDataCS.Dispatch(initKernel, voxTexSize / 8, voxTexSize / 8, voxTexSize / 8);
        Shader.SetGlobalInt("voxTexSize", voxTexSize);
        Shader.SetGlobalFloat("voxSize", voxSize);
        Shader.SetGlobalBuffer("VoxelTexture", voxelBuffer);
        // 初始化要用的矩阵
        LookUp = new Matrix4x4();
        LookForward = new Matrix4x4();
        LookRight = new Matrix4x4();
        OrthoTrans = Matrix4x4.Ortho(-voxAreaSize / 2, voxAreaSize / 2, 
                                -voxAreaSize / 2, voxAreaSize / 2, 
                                    0, voxAreaSize);
        //OrthoTrans = Camera.main.projectionMatrix;
        Shader.SetGlobalMatrix("VoxelAreaOrthoMat", GL.GetGPUProjectionMatrix(OrthoTrans, true));
        upMatIdx = Shader.PropertyToID("LookUpViewMat");
        forwardMatIdx = Shader.PropertyToID("LookForwardViewMat");
        rightMatIdx = Shader.PropertyToID("LookRightViewMat");
        camPosIdx = Shader.PropertyToID("manualCameraPos");
        // 跟踪主摄
        camTrans = Camera.main.transform;
    }

    void Update()
    {
        // 准备向量
        /*Vector3 unitZ = camTrans.forward;
        unitZ.y = 0;
        unitZ.Normalize();
        Vector3 unitX = Vector3.Cross(Vector3.up, unitZ);
        unitX.Normalize();
        Vector3 unitY = Vector3.up;
        unitY.Normalize();*/
        
        // 准备位置
        LookUpPos = LookForwardPos = LookRightPos = zeroPos;
        Vector3 alignedCamPos = zeroPos;
        alignedCamPos.x += voxAreaSize / 2;
        alignedCamPos.y += voxAreaSize / 2;
        alignedCamPos.z += voxAreaSize / 2;
        /*LookUpPos.y -= voxAreaSize / 2;
        LookForwardPos.z -= voxAreaSize / 2;
        LookRightPos.x -= voxAreaSize / 2;*/
        LookUpPos.x += voxAreaSize / 2; LookUpPos.z += voxAreaSize / 2;
        LookForwardPos.x += voxAreaSize / 2; LookForwardPos.y += voxAreaSize / 2;
        LookRightPos.y += voxAreaSize / 2; LookRightPos.z += voxAreaSize / 2;
        
        // 更改
        LookUpTrans.position = LookUpPos;
        LookForwardTrans.position = LookForwardPos;
        LookRightTrans.position = LookRightPos;
        LookUpTrans.LookAt(alignedCamPos, Vector3.forward);
        LookForwardTrans.LookAt(alignedCamPos, Vector3.up);
        LookRightTrans.LookAt(alignedCamPos, Vector3.up);
        
        // 准备矩阵
        LookUp = View(LookUpTrans);
        LookForward = View(LookForwardTrans);
        LookRight = View(LookRightTrans);
        
        Shader.SetGlobalMatrix(upMatIdx, LookUp);
        Shader.SetGlobalMatrix(forwardMatIdx, LookForward);
        Shader.SetGlobalMatrix(rightMatIdx, LookRight);
        Shader.SetGlobalVector(camPosIdx, camTrans.position);

        zeroPos = camTrans.position -
                  new Vector3(voxTexSize * voxSize / 2.0f, voxTexSize * voxSize / 2.0f,
                      voxTexSize * voxSize / 2.0f);
        float unit = voxSize * Mathf.Pow(2, lodLevels - 1);
        zeroPos.x = (int)(zeroPos.x / unit) * unit;
        zeroPos.y = (int)(zeroPos.y / unit) * unit;
        zeroPos.z = (int)(zeroPos.z / unit) * unit;
        Shader.SetGlobalVector("zeroPos", zeroPos);
        Shader.SetGlobalInt("_LodLevel", lodLevels);

        /*var viewMatrix = View(camTrans);
        Debug.Log("Cam:" + Camera.main.worldToCameraMatrix);
        Debug.Log("My: " + viewMatrix);*/
    }
    // 根据transform计算view矩阵（并非lookat）
    Matrix4x4 View(Transform tf)
    {
        var view = Matrix4x4.Rotate(Quaternion.Inverse(tf.rotation)) *
                   Matrix4x4.Translate(-tf.position);
        if (SystemInfo.usesReversedZBuffer)
        {
            view.m20 = -view.m20;
            view.m21 = -view.m21;
            view.m22 = -view.m22;
            view.m23 = -view.m23;
        }
        return view;
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(centerPos, voxAreaSize * Vector3.one);
    }
}

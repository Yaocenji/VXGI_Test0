using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class MotionVectorFeature : ScriptableRendererFeature
{
    [Header("运动向量RT")]
    public RenderTexture motionVectorRT;
    class MotionVectorRenderPass : ScriptableRenderPass
    {
        public RenderTexture motionVectorRT;
        public Camera mainCam;
        private Matrix4x4 lastMat_V;
        private Matrix4x4 lastMat_P;
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            ConfigureTarget(motionVectorRT);
            ConfigureClear(ClearFlag.All, Color.black);
        }
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("GetIndirectLight");
            
            context.SetupCameraProperties(renderingData.cameraData.camera);
            cmd.SetRenderTarget(motionVectorRT);
            context.ExecuteCommandBuffer(cmd);
            // 临时DrawSetting
            DrawingSettings drawSetting = new DrawingSettings(new ShaderTagId("MotionVector"),
                new SortingSettings(Camera.main));
            FilteringSettings fs = new FilteringSettings(RenderQueueRange.all);
            ScriptableCullingParameters cullingParameters;
            renderingData.cameraData.camera.TryGetCullingParameters(false, out cullingParameters);
            CullingResults cullResults = context.Cull(ref cullingParameters);
            
            context.DrawRenderers(cullResults, ref drawSetting, ref fs);

            
            // 更新VP矩阵
            lastMat_V = VoxelGI.View(mainCam.transform);
            lastMat_P = GL.GetGPUProjectionMatrix( Matrix4x4.Perspective(mainCam.fieldOfView, (float)(Screen.width) / (float)(Screen.height),
                mainCam.nearClipPlane, mainCam.farClipPlane), true);
            Shader.SetGlobalMatrix("LastFrameVPMat", lastMat_P * lastMat_V);
            
            cmd.Clear();
            cmd.Release();
        }
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
        }
    }

    MotionVectorRenderPass m_ScriptablePass;

    public override void Create()
    {
        m_ScriptablePass = new MotionVectorRenderPass();

        // Configures where the render pass should be injected.
        m_ScriptablePass.renderPassEvent = RenderPassEvent.BeforeRenderingOpaques;
        m_ScriptablePass.motionVectorRT = motionVectorRT;
        m_ScriptablePass.mainCam = Camera.main;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_ScriptablePass);
    }
}



using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Serialization;

public class VoxelMipmapRenderFeature : ScriptableRendererFeature
{
    // 体素范围：以摄像机为中心的正方体
    // 设置范围大小
    [Header("体素信息计算着色器")]
    public ComputeShader manageVoxelDataCS;

    class VoxelRenderPass : ScriptableRenderPass
    {
        public VoxelGI _vxgiManager;
        
        public ComputeShader manageVoxelDataCS;
        private int mipmapKernel;
        
        // 临时RT
        static string rt_name = "_VoxelTextureTarget";
        private static int rt_id = Shader.PropertyToID(rt_name);
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            /*RenderTextureDescriptor descriptor = new RenderTextureDescriptor(VoxelGI.instance.voxTexSize, VoxelGI.instance.voxTexSize, RenderTextureFormat.Default, 0);
            cmd.GetTemporaryRT(rt_id, descriptor);
            ConfigureTarget(rt_id);
            ConfigureClear(ClearFlag.All, Color.black);*/
            
            // 初始化CS
            mipmapKernel = manageVoxelDataCS.FindKernel("CalculateLOD");
            manageVoxelDataCS.SetBuffer(mipmapKernel, "VoxelTexture", VoxelGI.instance.voxelBuffer);
            manageVoxelDataCS.SetInt("voxTexSize", VoxelGI.instance.voxTexSize);
        }
        
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            // 临时DrawSetting
            //DrawingSettings drawSetting;
            //drawSetting = new DrawingSettings(new ShaderTagId("UniversalForward"),
                //new SortingSettings(Camera.main));
            //FilteringSettings fs = FilteringSettings.defaultValue;
            
            CommandBuffer cmd = CommandBufferPool.Get("mipmapVoxelData");
            int currVoxTexSize = VoxelGI.instance.voxTexSize / 2;
            int lastOffset = 0;
            int currOffset = VoxelGI.instance.voxTexSize * VoxelGI.instance.voxTexSize * VoxelGI.instance.voxTexSize;
            for (int i = 1; i < VoxelGI.instance.lodLevels; i++)
            {
                cmd.SetComputeBufferParam(manageVoxelDataCS, mipmapKernel, "VoxelTexture", VoxelGI.instance.voxelBuffer);
                cmd.SetComputeIntParam(manageVoxelDataCS, "lastOffset", lastOffset);
                cmd.SetComputeIntParam(manageVoxelDataCS, "offset", currOffset);
                cmd.SetComputeIntParam(manageVoxelDataCS, "voxTexSize", VoxelGI.instance.voxTexSize);
                cmd.SetComputeIntParam(manageVoxelDataCS, "currVoxTexSize", currVoxTexSize);
                cmd.SetComputeIntParam(manageVoxelDataCS, "currLodLevel", i);
                cmd.DispatchCompute(manageVoxelDataCS, mipmapKernel, currVoxTexSize / 8, currVoxTexSize / 8, currVoxTexSize / 8);
                
                lastOffset = currOffset;
                currOffset += currVoxTexSize * currVoxTexSize * currVoxTexSize;
                currVoxTexSize /= 2;
            }
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            cmd.Release();
        }

        // Cleanup any allocated resources that were created during the execution of this render pass.
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            //cmd.ReleaseTemporaryRT(rt_id);
            // do nothing
        }
    }

    VoxelRenderPass m_ScriptablePass;

    /// <inheritdoc/>
    public override void Create()
    {
        m_ScriptablePass = new VoxelRenderPass();

        // Configures where the render pass should be injected.
        m_ScriptablePass.renderPassEvent = RenderPassEvent.AfterRendering;

        m_ScriptablePass.manageVoxelDataCS = manageVoxelDataCS;
    }
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_ScriptablePass);
    }
}



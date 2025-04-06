Shader "LearnURP/ConeTracing"
{
    Properties
    {
        _BaseMap ("Base Texture",2D) = "white"{}
        _BaseColor("Base Color",Color)=(1,1,1,1)
        _SpecularColor("Specular Color",Color)=(1,1,1,1)
        _Smoothness("Smoothness", Float) = 0.5
        // _VoxelData("Voxel Data", 3D) = "black"{}
    }
    SubShader
    {
        Tags { 
            "RenderPipeline" = "UniversalPipeline"
            "Queue"="Geometry"
            "RenderType"="Opaque" 
        }
        HLSLINCLUDE

        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
        
        // 场景体素buffer
        //除了贴图外，要暴露在Inspector面板上的变量都需要缓存到CBUFFER中
        /*struct VoxelData{
            float3 col;
        };*/
        
        #include "Assets/LearnURP/VXGI/voxelData.hlsl"
        
        CBUFFER_START(UnityPerMaterial)
        float4 _BaseMap_ST;
        half4 _BaseColor;
        half4 _SpecularColor;
        half _Smoothness;
        int voxTexSize; // 体素网格边长（一般是256）
        float voxSize;    // 体素大小边长 默认是0.125 1/8
        matrix<float, 4, 4> LookUpViewMat;
        matrix<float, 4, 4> LookForwardViewMat;
        matrix<float, 4, 4> LookRightViewMat;
        matrix<float, 4, 4> VoxelAreaOrthoMat;
        float3 manualCameraPos;
        //matrix<float, 4, 4> TESTMAT;
        CBUFFER_END
        
        ENDHLSL
        
        Pass
        {
            Tags{
                "LightMode" = "UniversalForward"
            }
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float4 tangentOS : TANGENT;
                float2 uv : TEXCOORD0;
            };

            struct Varings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float3 positionWS : TEXCOORD3;
                float4 tangentWS : TEXCOORD4;
            };

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            RWStructuredBuffer<VoxelData> VoxelTexture : register(u1); // 表面是1维，实际是三维

            // 顶点着色器
            Varings vert (Attributes IN)
            {
                Varings OUT;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(IN.normalOS.xyz, IN.tangentOS.xyzw);
                OUT.positionCS = positionInputs.positionCS;
                OUT.viewDirWS = GetCameraPositionWS() - positionInputs.positionWS;
                OUT.normalWS = normalInputs.normalWS;
                OUT.uv=TRANSFORM_TEX(IN.uv,_BaseMap);
                OUT.positionWS = positionInputs.positionWS;
                OUT.tangentWS.xyz = normalInputs.tangentWS;
                OUT.tangentWS.w = IN.tangentOS.w;
                return OUT;
            }
            // 片元着色器
            float4 frag (Varings IN) : SV_Target
            {
                // 根据世界空间位置读取VoxelTexrure
                // 计算VoxelTexture所需id
                uint3 id = getId(manualCameraPos, IN.positionWS, voxTexSize, voxSize);
                VIS_VOX_IDX(id);
                
                // 先进行正常的渲染
                // sample the texture and color
                half4 baseMap = Gamma22ToLinear(SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv));
                half4 specularColor = Gamma22ToLinear(_SpecularColor);
                
                Light light = GetMainLight();
                float3 lightDirWS = light.direction;
                float3 viewDirWS = GetCameraPositionWS() - IN.positionWS;

                half3 diffuse = baseMap.xyz*_BaseColor*LightingLambert(light.color, light.direction, IN.normalWS);
                
                float3 halfVec = SafeNormalize(normalize(lightDirWS) + normalize(viewDirWS));
                half NdotH = saturate(dot(normalize(IN.normalWS), halfVec));
                half3 specular = light.color * specularColor.rgb * pow(NdotH, _Smoothness);

                // 计算圆锥体radiance
                // 7个60°圆锥体，两两相切组成

                // 先获取切线空间
                // 切线空间到世界空间的变换矩阵
                real3x3 T2W_MATRIX = CreateTangentToWorld(IN.normalWS, IN.tangentWS.xyz, IN.tangentWS.w);
                //切线空间采样方向（七个圆锥体的中心向量）
                float3 dir[7];
                dir[0] = float3(0, 1, 0);
                dir[1] = normalize(float3(3, 2, 1.73205));
                dir[2] = normalize(float3(0, 1, 1.73205));
                dir[3] = normalize(float3(-3, 2, 1.73205));
                dir[4] = normalize(float3(-3, 2, -1.73205));
                dir[5] = normalize(float3(0, 1, -1.73205));
                dir[6] = normalize(float3(3, 2, -1.73205));
                // 将其变换为：世界空间
                dir[0] = TransformTangentToWorld(dir[0], T2W_MATRIX);
                dir[1] = TransformTangentToWorld(dir[1], T2W_MATRIX);
                dir[2] = TransformTangentToWorld(dir[2], T2W_MATRIX);
                dir[3] = TransformTangentToWorld(dir[3], T2W_MATRIX);
                dir[4] = TransformTangentToWorld(dir[4], T2W_MATRIX);
                dir[5] = TransformTangentToWorld(dir[5], T2W_MATRIX);
                dir[6] = TransformTangentToWorld(dir[6], T2W_MATRIX);

                // 沿着七个圆锥方向步进
                float3 marchPos[7];
                for (int i = 0; i < 7; ++i){
                    marchPos[i] = IN.positionWS;    // 初始化
                    // 步进
                    // 初始步进距离：根号3 / 2 * 最小体素边长
                    // 每次步进都将步进距离翻倍
                    float marchDist = 1.73205 / 2.0 * voxSize;
                    // 每次步进，访问LOD级别都增大
                    int lodLevel = 0;
                    /*for (int i = 0; i < 5; ++i)
                    {
                        
                    }*/
                }
                
                float3 color = VoxelTexture[visIdxLODById(voxTexSize, 2, id)].col;

                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
        UsePass "Universal Render Pipeline/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Lit/DepthNormals"
    }
}

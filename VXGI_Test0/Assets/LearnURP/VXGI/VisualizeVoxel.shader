Shader "LearnURP/VisualizeVoxel"
{
    Properties
    {
        _BaseMap ("Base Texture",2D) = "white"{}
        _BaseColor("Base Color",Color)=(1,1,1,1)
        _SpecularColor("Specular Color",Color)=(1,1,1,1)
        _Smoothness("Smoothness", Float) = 0.5
        _LodLevel("Displayed Lod Level", Int) = 2
        _Visualize("Enable Visualize Voxel", Int) = 1
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
        
        #include "Assets/LearnURP/VXGI/voxelData.hlsl"
        
        CBUFFER_START(UnityPerMaterial)
        float4 _BaseMap_ST;
        half4 _BaseColor;
        half4 _SpecularColor;
        half _Smoothness;
        int voxTexSize; // 体素网格边长（一般是256）
        float voxSize;    // 体素大小边长 默认是0.125 1/8
        float3 manualCameraPos;
        int _LodLevel;   // 当前采样的lod级别
        int _Visualize;
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
            #pragma geometry geom
            #pragma fragment frag

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD;
            };

            struct VertexToGeometry
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float3 positionWS : TEXCOORD3;
            };

            struct Varings
            {
                float4 positionCS : SV_POSITION;
                float3 viewDirWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float4 positionWS : TEXCOORD3;  // xyz表示世界空间位置，w表示世界空间深度（基于对应的正交投影面）
                float3 NormalizedID : TEXCOORD4;
            };

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            RWStructuredBuffer<VoxelData> VoxelTexture : register(u1); // 表面是1维，实际是三维

            // 顶点着色器
            VertexToGeometry vert (Attributes IN)
            {
                VertexToGeometry OUT;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(IN.normalOS.xyz);
                OUT.positionCS = positionInputs.positionCS;
                OUT.viewDirWS = GetCameraPositionWS() - positionInputs.positionWS;
                OUT.normalWS = normalInputs.normalWS;
                OUT.uv=TRANSFORM_TEX(IN.uv,_BaseMap);
                OUT.positionWS = positionInputs.positionWS;
                return OUT;
            }

            #define APPEND_POINT(i)                                                 \
                    OUT.positionCS = TransformWorldToHClip(float4(points[i], 1.0)); \
                    OUT.viewDirWS = GetCameraPositionWS() - points[i];              \
                    OUT.positionWS = float4(points[i], 1.0f);                       \
                    triStream.Append(OUT);                                          
            
            // 几何着色器
            [maxvertexcount(54)]
            void geom (triangle VertexToGeometry IN[3], inout TriangleStream<Varings> triStream)
            {
                if (_Visualize == 0) return;
                for (int i = 0; i < 3 ; i ++)
                {
                    uint3 id = getId(manualCameraPos, IN[i].positionWS, voxTexSize, voxSize);
                    VIS_VOX_IDX(id);
                    if (VoxelTexture[visId].flags.x == 0) continue;
                    Varings OUT;
                    OUT.NormalizedID = (float3(id) + 0.2) / float(voxTexSize);
                    float3 points[8];
                    points[0] = IN[i].positionWS;
                    points[1] = IN[i].positionWS + float3(1, 0, 0) * voxSize;
                    points[2] = IN[i].positionWS + float3(1, 0, 1) * voxSize;
                    points[3] = IN[i].positionWS + float3(0, 0, 1) * voxSize;

                    points[4] = IN[i].positionWS + float3(0, 1, 0) * voxSize;
                    points[5] = IN[i].positionWS + float3(1, 1, 0) * voxSize;
                    points[6] = IN[i].positionWS + float3(1, 1, 1) * voxSize;
                    points[7] = IN[i].positionWS + float3(0, 1, 1) * voxSize;

                    if (dot(float3(0, 0, 1), IN[0].viewDirWS) < 0) {
                        // 正面两个三角形
                        OUT.normalWS = float3(0, 0, -1);
                        // 045
                        APPEND_POINT(0)
                        APPEND_POINT(4)
                        APPEND_POINT(5)
                    triStream.RestartStrip();
                        // 051
                        APPEND_POINT(0)
                        APPEND_POINT(5)
                        APPEND_POINT(1)
                    triStream.RestartStrip();
                    } else {
                        // 背面两个三角形
                        OUT.normalWS = float3(0, 0, 1);
                        // 267
                        APPEND_POINT(2)
                        APPEND_POINT(6)
                        APPEND_POINT(7)
                    triStream.RestartStrip();
                        // 273
                        APPEND_POINT(2)
                        APPEND_POINT(7)
                        APPEND_POINT(3)
                    triStream.RestartStrip();
                    }
                    if (dot(float3(1, 0, 0), IN[0].viewDirWS) < 0) {
                        // 左面两个三角形
                        OUT.normalWS = float3(-1, 0, 0);
                        // 374
                        APPEND_POINT(3)
                        APPEND_POINT(7)
                        APPEND_POINT(4)
                    triStream.RestartStrip();
                        // 340
                        APPEND_POINT(3)
                        APPEND_POINT(4)
                        APPEND_POINT(0)
                    triStream.RestartStrip();
                    } else {
                        // 右面两个三角形
                        OUT.normalWS = float3(1, 0, 0);
                        // 156
                        APPEND_POINT(1)
                        APPEND_POINT(5)
                        APPEND_POINT(6)
                    triStream.RestartStrip();
                        // 162
                        APPEND_POINT(1)
                        APPEND_POINT(6)
                        APPEND_POINT(2)
                    triStream.RestartStrip();
                    }
                    if (dot(float3(0, 1, 0), IN[0].viewDirWS) < 0) {
                        // 底面两个三角形
                        OUT.normalWS = float3(0, -1, 0);
                        // 301
                        APPEND_POINT(3)
                        APPEND_POINT(0)
                        APPEND_POINT(1)
                    triStream.RestartStrip();
                        // 312
                        APPEND_POINT(3)
                        APPEND_POINT(1)
                        APPEND_POINT(2)
                    triStream.RestartStrip();
                    } else {
                        // 顶面两个三角形
                        OUT.normalWS = float3(0, 1, 0);
                        // 476
                        APPEND_POINT(4)
                        APPEND_POINT(7)
                        APPEND_POINT(6)
                    triStream.RestartStrip();
                        // 465
                        APPEND_POINT(4)
                        APPEND_POINT(6)
                        APPEND_POINT(5)
                    triStream.RestartStrip();
                    }

                }
            }
            // 片元着色器
            float4 frag (Varings IN) : SV_Target
            {
                // 根据世界空间位置写入VoxelTexrure
                // 计算VoxelTexture所需id
                /*float3 zeroPos = manualCameraPos - (voxTexSize * voxSize / 2.0f).xxx;
                //float3 id = (IN.positionWS - zeroPos) / voxSize;*/
                uint3 id = getId(manualCameraPos, IN.positionWS, voxTexSize, voxSize);
                id = IN.NormalizedID * voxTexSize;

                int visId = id.x * voxTexSize * voxTexSize + id.y * voxTexSize + id.z;
                
                int curTexSize = voxTexSize / 4;
                uint3 curID = id / 4;
                int tmpSize = voxTexSize * voxTexSize * voxTexSize;
                visId = tmpSize + tmpSize / 8 + curID.x * curTexSize * curTexSize + curID.y * curTexSize + curID.z;
                
                //color = diffuse * 0.9 + 0.1;
                curTexSize = voxTexSize / int(pow(2, _LodLevel));
                curID = id / int(pow(2, _LodLevel));
                visId = int(tmpSize * (8 - pow(0.125, _LodLevel - 1))) / 7;
                visId += curID.x * curTexSize * curTexSize + curID.y * curTexSize + curID.z;
                
                float3 color = VoxelTexture[visIdxLODById(voxTexSize, _LodLevel, id)].col;
                
                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
        UsePass "Universal Render Pipeline/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Lit/DepthNormals"
    }
}

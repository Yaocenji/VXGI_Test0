Shader "LearnURP/Voxelization"
{
    Properties
    {
        _BaseMap ("Base Texture",2D) = "white"{}
        _BaseColor("Base Color",Color)=(1,1,1,1)
        _SpecularColor("Specular Color",Color)=(1,1,1,1)
        _Smoothness("Smoothness", Float) = 0.5
        _EmitionCol("Emit Color",Color)=(0,0, 0, 0)
        _EmitionStrength("Emit Strength", Float) = 1.0
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
        
        
        #include "Assets/LearnURP/VXGI/voxelData.hlsl"
        
        CBUFFER_START(UnityPerMaterial)
        float4 _BaseMap_ST;
        half4 _BaseColor;
        half4 _SpecularColor;
        half _Smoothness;
        half4 _EmitionCol;
        float _EmitionStrength;
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
            Cull Off
            ZTest Always
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma geometry geom
            #pragma fragment frag

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN      // URP 主光阴影、联机阴影、屏幕空间阴影
            #pragma multi_compile_fragment _ _SHADOWS_SOFT      // URP 软阴影

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
                float4 shadowCoord : TEXCOORD4;
            };

            struct Varings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float4 positionWS : TEXCOORD3;  // xyz表示世界空间位置，w表示世界空间深度（基于对应的正交投影面）
                float4 shadowCoord : TEXCOORD4;
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
                OUT.shadowCoord = GetShadowCoord(positionInputs);
                return OUT;
            }
            // 几何着色器
            [maxvertexcount(3)]
            void geom (triangle VertexToGeometry IN[3], inout TriangleStream<Varings> triStream)
            {
                Varings OUT;
                // 计算法线、选择方向和对应的投影方向
                half3 triNormal =   // 单位化的面法线
                    normalize(cross(IN[0].positionWS - IN[1].positionWS, IN[0].positionWS - IN[2].positionWS));
                // 计算面法线和xyz轴单位向量点积
                half dotX = abs(dot(triNormal, half3(1, 0, 0)));
                half dotY = abs(dot(triNormal, half3(0, 1, 0)));
                half dotZ = abs(dot(triNormal, half3(0, 0, 1)));
                MATRIX viewMat;
                int projDir;    // 投影方向
                if (dotX >= dotY && dotX >= dotZ) {
                    viewMat = LookRightViewMat;
                    projDir = 0;
                }else if (dotY >= dotX && dotY >= dotZ) {
                    viewMat = LookUpViewMat;
                    projDir = 1;
                } else // dotZ >= dotX && dotZ >= dotY
                {
                    viewMat = LookForwardViewMat;
                    projDir = 2;
                }
                for (int i = 0; i < 3; i++)
                {
                    OUT.positionCS = mul(mul(VoxelAreaOrthoMat, viewMat), float4(IN[i].positionWS, 1.0));
                    //OUT.positionCS = TransformWorldToHClip(float4(IN[i].positionWS, 1.0));
                    OUT.normalWS = IN[i].normalWS;
                    OUT.viewDirWS = IN[i].viewDirWS;
                    OUT.uv = IN[i].uv;
                    OUT.positionWS = float4(IN[i].positionWS, 0.0f);
                    OUT.shadowCoord = IN[i].shadowCoord;

                    // 计算深度
                    if (projDir == 0)
                        OUT.positionWS.w = OUT.positionWS.x - (manualCameraPos.x - voxTexSize * voxSize / 2.0f);
                    else if (projDir == 1)
                        OUT.positionWS.w = OUT.positionWS.y - (manualCameraPos.y - voxTexSize * voxSize / 2.0f);
                    else    // if (projDir == 2)
                        OUT.positionWS.w = OUT.positionWS.z - (manualCameraPos.z - voxTexSize * voxSize / 2.0f);
                    //OUT.positionWS.w = (float)projDir / 3.0f;
                    
                    triStream.Append(OUT);
                }
            }

            // 片元着色器
            float4 frag (Varings IN) : SV_Target
            {
                // 先进行正常的渲染
                // sample the texture and color
                half4 baseMap = (SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv));
                half4 specularColor = (_SpecularColor);
                
                Light light = GetMainLight();
                float3 lightDirWS = light.direction;
                float3 viewDirWS = GetCameraPositionWS() - IN.positionWS;

                half3 diffuse = baseMap.xyz*_BaseColor*LightingLambert(light.color, light.direction, IN.normalWS);
                
                float3 halfVec = SafeNormalize(normalize(lightDirWS) + normalize(viewDirWS));
                half NdotH = saturate(dot(normalize(IN.normalWS), halfVec));
                half3 specular = light.color * specularColor.rgb * pow(NdotH, _Smoothness);
                
                float3 color = diffuse + specular;
                float shadow = MainLightRealtimeShadow(IN.shadowCoord);
                color *= shadow;
                // 自发光
                color += _EmitionCol * _EmitionStrength;

                // 根据世界空间位置写入VoxelTexrure
                // 计算VoxelTexture所需id
                uint3 id = getId(manualCameraPos, IN.positionWS, voxTexSize, voxSize);
                VIS_VOX_IDX(id);

                VoxelTexture[visitVoxIndex(id, voxTexSize)].col = color;
                VoxelTexture[visitVoxIndex(id, voxTexSize)].flags.x = 1;
                
                // debug
                //VoxelTexture[visitVoxIndex(id, voxTexSize)].col = IN.positionWS / (voxTexSize * voxSize);
                
                clip(-1);
                
                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
        UsePass "Universal Render Pipeline/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Lit/DepthNormals"
        UsePass "Universal Render Pipeline/Lit/ShadowCaster"    // 产生投影
    }
}

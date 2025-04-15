Shader "LearnURP/ConeTracing"
{
    Properties
    {
        _BaseMap ("Base Texture",2D) = "white"{}
        _BaseColor("Base Color",Color)=(1,1,1,1)
        _Roughness ("Roughness Texture",2D) = "black"{}
        _Normal ("Normal Texture",2D) = "Bump"{}
        _NormalScale ("Normal Scale", Float) = 1.0
        _Metallic ("Metallic Texture",2D) = "white"{}
        _AmbientOcclusion ("Ambient Occlusion Texture",2D) = "white"{}
        _SpecularColor("Specular Color",Color)=(1,1,1,1)
        _Smoothness("Smoothness", Float) = 0.5
        _EmitionCol("Emit Color",Color)=(0,0, 0, 0)
        _EmitionStrength("Emit Strength", Float) = 1.0
        _WhiteNoise0("White Noise 0", 2D) = "white"{}
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
        float4 _Roughness_ST;
        float4 _Normal_ST;
        float _NormalScale;
        float4 _Metallic_ST;
        float4 _AmbientOcclusion_ST;
        
        half4 _SpecularColor;
        half _Smoothness;
        half4 _EmitionCol;
        float _EmitionStrength;
        float4 _WhiteNoise0_ST;
        
        int voxTexSize; // 体素网格边长（一般是256）
        float voxSize;    // 体素大小边长 默认是0.125 1/8
        matrix<float, 4, 4> LookUpViewMat;
        matrix<float, 4, 4> LookForwardViewMat;
        matrix<float, 4, 4> LookRightViewMat;
        matrix<float, 4, 4> VoxelAreaOrthoMat;
        float3 manualCameraPos;
        //matrix<float, 4, 4> TESTMAT;

        int _LodLevel;
        CBUFFER_END


        // 以下是PBR轮子
        float Pow5(float x){
            return x * x * x * x * x;
        }
        float Pow2(float x){
            return x * x;
        }
        // 菲涅尔
        float3 F_SchlickFresnel(float cosTheta, float3 F0){
            return F0 + (1 - F0) * Pow5(1 - cosTheta);
        }
        // 法线分布
        float D_GGX(float3 n, float3 h, float rough){
            float a2 = Pow2(rough);
            float NdotH = dot(n, h);
            float denom = PI * Pow2(Pow2(NdotH) * (a2 - 1) + 1);
            return a2 / denom;
        }
        // 几何遮蔽函数
        float G_SchlickGGX(float3 n, float3 v, float3 rough, bool if_IBL = false){
            float k;
            if (if_IBL){
                k = Pow2(rough) / 2;
            }
            else {
                k = Pow2(rough + 1) / 8;
            }
            float NdotV = max(dot(n, v), 0) + 0.001;
            return NdotV / (NdotV * (1 - k) + k);
        }
        // 几何遮蔽函数史密斯法：观察方向的几何遮蔽函数和光照方向的几何遮蔽函数相乘获得G项
        float G_Smith(float3 n, float3 v, float3 l, float3 rough, bool if_IBL = false){
            return G_SchlickGGX(n, v, rough, if_IBL) * G_SchlickGGX(n, l, rough, if_IBL);
        }


        float3 BRDF(float3 L, float3 V, float3 N, half3 baseColor, half3 metalness, half3 roughness, bool if_IBL = false){
            // 计算specular
            float3 H = normalize(L + V);
            float D = D_GGX(N, H, roughness);
            float G = G_Smith(N, V, L, roughness, if_IBL);

            float3 F0 = float3(0.04f, 0.04f, 0.04f);
            F0 = lerp(F0, baseColor, metalness.r);
            float3 F = F_SchlickFresnel(max(dot(H, V), 0), F0);

            float3 nom = D * G * F;
            float donom = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
            float3 Cook_Torrance = nom / donom;

            float3 Ks = F;
            float3 BRDF_Specular = Cook_Torrance;

            // 计算diffuse
            float3 Kd = float3(1,1,1) - Ks;
            Kd *= 1 - metalness;
            float3 BRDF_Diffuse = Kd * baseColor / PI;
            

            return BRDF_Specular + BRDF_Diffuse;
        }

//D项
float DistributionGGX(float3 NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
//G项
float GeometrySchlickGGX(float cosTheta, float k)
{
    float nom = cosTheta;
    float denom = cosTheta * (1.0 - k) + k;
    return nom / (denom + 1e-5f);
}
//G项考虑视线方向和光照方向
float GeometrySmith(float NdotV,float NdotL ,float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float ggx2 = GeometrySchlickGGX(NdotV, k);
    float ggx1 = GeometrySchlickGGX(NdotL, k);
    return ggx1 * ggx2;
}
//F项
float3 FresnelTerm(float3 F0, float cosA)
{
    half t = pow(1 - cosA,5.0); 
    return F0 + (1 - F0) * t;
}


//直接光照计算
float3 DirectPBR(float nl,float nv,float nh,float hv,float3 albedo,float metalness,float roughness,float3 f0,float3 lightColor)
{
    float dTerm = DistributionGGX(nh, roughness);
    float gTerm = GeometrySmith(nl, nv, roughness);
    float3 fTerm = FresnelTerm(f0, hv);
    //max 0.001 保证分母不为0
    float3 specular = dTerm * gTerm * fTerm / (4.0 * max(nv * nl, 0.001));
    //我们按照能量守恒的关系，首先计算镜面反射部分，它的值等于入射光线被反射的能量所占的百分比。
    float3 kS = fTerm;
    //然后折射光部分就可以直接由镜面反射部分计算得出：
    float3 kD = (1.0 - kS) ;
    //金属是没有漫反射的,所以Kd需要乘上1-metalness
    kD *= 1.0 - metalness;
    //除π是为了能量守恒，但Unity也没有除以π，应该是觉得除以π后太暗，所以我们这里也先不除
    float3 diffuse = kD * albedo;// *INV_PI;
    float3 result = (diffuse + specular) * nl * lightColor;
    return (result);
}
        
        float Clamp(float x, float l, float r)
        {
            if (x > r)
                return r;
            if (x < l)
                return l;
            return x;
        }

        
        TEXTURE2D(_BaseMap); SAMPLER(sampler_BaseMap);
        TEXTURE2D(_Roughness); SAMPLER(sampler_Roughness);
        TEXTURE2D(_Normal); SAMPLER(sampler_Normal);
        TEXTURE2D(_Metallic); SAMPLER(sampler_Metallic);
        TEXTURE2D(_AmbientOcclusion); SAMPLER(sampler_AmbientOcclusion);
        
        TEXTURE2D(_WhiteNoise0); SAMPLER(sampler_WhiteNoise0);
        RWStructuredBuffer<VoxelData> VoxelTexture : register(u1); // 表面是1维，实际是三维


        float3 GetAlbedo(float2 uv)
        {
            return SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, uv).xyz * _BaseColor;
        }
        float3 GetRoughness(float2 uv)
        {
            return SAMPLE_TEXTURE2D(_Roughness, sampler_Roughness, uv).xyz;
        }
        float3 GetNormalTS(float2 uv)
        {
            float4 packedNormal = SAMPLE_TEXTURE2D(_Normal, sampler_Normal, uv);
            float3 normalTS = UnpackNormal(packedNormal);
            normalTS.xy *= _NormalScale;
            normalTS.z = sqrt(1.0 - saturate(dot(normalTS.xy, normalTS.xy)));
            return normalTS;
        }
        float3 GetMetallic(float2 uv)
        {
            return SAMPLE_TEXTURE2D(_Metallic, sampler_Metallic, uv);
        }
        float GetAO(float2 uv)
        {
            return SAMPLE_TEXTURE2D(_AmbientOcclusion, sampler_AmbientOcclusion, uv).x;
        }
        
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
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN      // URP 主光阴影、联机阴影、屏幕空间阴影
            #pragma multi_compile_fragment _ _SHADOWS_SOFT      // URP 软阴影

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
                float4 shadowCoord : TEXCOORD5;
            };

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
                OUT.shadowCoord = GetShadowCoord(positionInputs);
                return OUT;
            }
            // 片元着色器
            float4 frag (Varings IN) : SV_Target
            {
                // 根据世界空间位置读取VoxelTexrure
                // 计算VoxelTexture所需id
                uint3 id = getId(GetCameraPositionWS(), IN.positionWS, voxTexSize, voxSize);
                VIS_VOX_IDX(id);
                
                // 先进行正常的渲染
                // sample the texture and color
                half4 baseMap = (SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv));
                half4 specularColor = (_SpecularColor);
                
                Light light = GetMainLight();
                float3 lightDirWS = light.direction;
                float3 viewDirWS = GetCameraPositionWS() - IN.positionWS;

                half3 albedo = baseMap.xyz*_BaseColor;

                half3 diffuse = albedo * LightingLambert(light.color, light.direction, IN.normalWS);
                
                float3 halfVec = SafeNormalize(normalize(lightDirWS) + normalize(viewDirWS));
                half NdotH = saturate(dot(normalize(IN.normalWS), halfVec));
                half3 specular = light.color * specularColor.rgb * pow(NdotH, _Smoothness);

                float3 color = diffuse;// + specular;

                // PBR，启动！
                // albedo
                float3 baseColor = GetAlbedo(IN.uv);
                // 粗糙度
                float3 roughness = GetRoughness(IN.uv);
                // 法线
                float3 normalTS = GetNormalTS(IN.uv);
                // 金属度
                float3 metallic = GetMetallic(IN.uv);
                // AO
                float ambientOcclusion = GetAO(IN.uv);

                // 变换到切线空间计算
                real3x3 T2W_MATRIX = CreateTangentToWorld(IN.normalWS, IN.tangentWS.xyz, IN.tangentWS.w);
                real3x3 W2T_MATRIX = Inverse(real4x4(
                                                        float4(T2W_MATRIX[0], 0),
                                                        float4(T2W_MATRIX[1], 0),
                                                        float4(T2W_MATRIX[2], 0),
                                                        float4(0,   0,   0,   1)));
                float3 lightDirTS = normalize(mul(light.direction, W2T_MATRIX));
                float3 viewDirTS = normalize(mul(IN.viewDirWS, W2T_MATRIX));
                float3 halfDirTS = normalize(lightDirTS + viewDirTS);

                // 计算BRDF颜色
                float3 brdf = BRDF(lightDirTS,viewDirTS, normalTS, baseColor, metallic, roughness);
                // 渲染方程
                float3 Lo = light.distanceAttenuation * light.color.rgb * max(dot(lightDirTS, normalTS), 0) * brdf;

                /*Lo = DirectPBR(dot(normalTS, lightDirTS), dot(normalTS, viewDirTS), dot(normalTS, halfDirTS), dot(halfDirTS, viewDirTS),
                    baseColor, metallic, roughness, float3(0.04f, 0.04f, 0.04f), light.color);*/
                color = Lo;

                // 阴影
                float shadow = MainLightRealtimeShadow(TransformWorldToShadowCoord(IN.positionWS));
                color *= shadow;

                // 自发光
                color += _EmitionCol * _EmitionStrength;


                // debug 
                //color = VoxelTexture[visId].col;
                //color = dir[0] * 0.5 + 0.5;
                //color = mul(IN.normalWS, W2T_MATRIX);

                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
        
        Pass
        {
            Tags{
                "LightMode" = "Voxelization"
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
                float4 tangentOS : TANGENT;
            };

            struct VertexToGeometry
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float3 positionWS : TEXCOORD3;
                float4 shadowCoord : TEXCOORD4;
                float4 tangentWS : TEXCOORD5;
            };

            struct Varings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
                float4 positionWS : TEXCOORD3;  // xyz表示世界空间位置，w表示世界空间深度（基于对应的正交投影面）
                float4 shadowCoord : TEXCOORD4;
                float4 tangentWS : TEXCOORD5;
            };

            // 顶点着色器
            VertexToGeometry vert (Attributes IN)
            {
                VertexToGeometry OUT;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(IN.normalOS.xyz, IN.tangentOS.xyzw);
                OUT.positionCS = positionInputs.positionCS;
                OUT.viewDirWS = GetCameraPositionWS() - positionInputs.positionWS;
                OUT.normalWS = normalInputs.normalWS;
                OUT.uv=TRANSFORM_TEX(IN.uv,_BaseMap);
                OUT.positionWS = positionInputs.positionWS;
                OUT.shadowCoord = GetShadowCoord(positionInputs);
                OUT.tangentWS.xyz = normalInputs.tangentWS;
                OUT.tangentWS.w = IN.tangentOS.w;
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
                    OUT.tangentWS = IN[i].tangentWS;

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
                // 先进行正常的渲染
                // sample the texture and color
                half4 baseMap = (SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv));
                half4 specularColor = (_SpecularColor);
                
                Light light = GetMainLight();
                float3 lightDirWS = light.direction;
                float3 viewDirWS = GetCameraPositionWS() - IN.positionWS;

                half3 albedo = baseMap.xyz*_BaseColor;

                half3 diffuse = albedo * LightingLambert(light.color, light.direction, IN.normalWS);
                
                float3 halfVec = SafeNormalize(normalize(lightDirWS) + normalize(viewDirWS));
                half NdotH = saturate(dot(normalize(IN.normalWS), halfVec));
                half3 specular = light.color * specularColor.rgb * pow(NdotH, _Smoothness);

                float3 color = diffuse;// + specular;

                
                // PBR，启动！
                // albedo
                float3 baseColor = GetAlbedo(IN.uv);
                // 粗糙度
                float3 roughness = GetRoughness(IN.uv);
                // 法线
                float3 normalTS = GetNormalTS(IN.uv);
                // 金属度
                float3 metallic = GetMetallic(IN.uv);

                // 变换到切线空间计算
                real3x3 T2W_MATRIX = CreateTangentToWorld(IN.normalWS, IN.tangentWS.xyz, IN.tangentWS.w);
                real3x3 W2T_MATRIX = Inverse(real4x4(
                                                        float4(T2W_MATRIX[0], 0),
                                                        float4(T2W_MATRIX[1], 0),
                                                        float4(T2W_MATRIX[2], 0),
                                                        float4(0,   0,   0,   1)));
                float3 lightDirTS = normalize(mul(light.direction, W2T_MATRIX));
                float3 viewDirTS = normalize(mul(IN.viewDirWS, W2T_MATRIX));
                float3 halfDirTS = normalize(lightDirTS + viewDirTS);

                // 计算BRDF颜色
                float3 brdf = BRDF(lightDirTS,viewDirTS, normalTS, baseColor, metallic, roughness);
                // 渲染方程
                float3 Lo = light.distanceAttenuation * light.color.rgb * max(dot(lightDirTS, normalTS), 0) * brdf;

                /*Lo = DirectPBR(dot(normalTS, lightDirTS), dot(normalTS, viewDirTS), dot(normalTS, halfDirTS), dot(halfDirTS, viewDirTS),
                    baseColor, metallic, roughness, float3(0.04f, 0.04f, 0.04f), light.color);*/
                color = Lo;

                
                float shadow = MainLightRealtimeShadow(TransformWorldToShadowCoord(IN.positionWS));
                color *= shadow;
                // 自发光
                color += _EmitionCol * _EmitionStrength;

                // 根据世界空间位置写入VoxelTexrure
                // 计算VoxelTexture所需id
                uint3 id = getId(manualCameraPos, IN.positionWS, voxTexSize, voxSize);
                VIS_VOX_IDX(id);

                int curr_flag_x = VoxelTexture[visitVoxIndex(id, voxTexSize)].flags.x;
                
                VoxelTexture[visitVoxIndex(id, voxTexSize)].col =  (color + curr_flag_x * VoxelTexture[visitVoxIndex(id, voxTexSize)].col) / float(curr_flag_x + 1);
                
                VoxelTexture[visitVoxIndex(id, voxTexSize)].norm = normalize(normalize(IN.normalWS) + curr_flag_x * VoxelTexture[visitVoxIndex(id, voxTexSize)].norm);
                
                VoxelTexture[visitVoxIndex(id, voxTexSize)].flags.x += 1;
                
                // debug
                //VoxelTexture[visitVoxIndex(id, voxTexSize)].col = IN.positionWS / (voxTexSize * voxSize);
                
                //clip(-1);
                
                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
        
        Pass
        {
            Tags{
                "LightMode" = "IndirectLight"
            }
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN      // URP 主光阴影、联机阴影、屏幕空间阴影
            #pragma multi_compile_fragment _ _SHADOWS_SOFT      // URP 软阴影

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
                float4 shadowCoord : TEXCOORD5;
            };

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
                OUT.shadowCoord = GetShadowCoord(positionInputs);
                return OUT;
            }

            // 噪声，每帧调用结果都不同
            float2 InterleavedGradientNoise2D(float2 v) {
                return float2(
                    frac(52.9829189f * frac((v.x * 0.6711056f + v.y * 0.0583715f) * frac(v.y * 0.6558879f + (_Time.z) * 0.0583715f))),
                    frac(19.5246817f * frac((v.y * 0.8798958f + v.x * 0.4791113579f) * frac(v.x * 0.6558879f + (_Time.z) * 0.0583715f)))
                    );
            }
            float2 InterleavedGradientNoise3D(float3 v) {
                return float2(
                    frac(52.9829189f * frac((v.x * 0.6711056f + v.y * 0.0583715f) * frac(v.z * 0.6558879f + (_Time.z) * 0.0583715f))),
                    frac(19.5246817f * frac((v.y * 0.8798958f + v.z * 0.4791113579f) * frac(v.x * 0.6558879f + (_Time.z) * 0.0583715f)))
                    );
            }
            // 片元着色器
            float4 frag (Varings IN) : SV_Target
            {
                // 根据世界空间位置读取VoxelTexrure
                // 计算VoxelTexture所需id
                uint3 id = getId(GetCameraPositionWS(), IN.positionWS, voxTexSize, voxSize);
                VIS_VOX_IDX(id);
                
                // 先进行正常的渲染
                // sample the texture and color
                half4 baseMap = (SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv));
                half4 specularColor = (_SpecularColor);
                
                Light light = GetMainLight();
                float3 lightDirWS = light.direction;
                float3 viewDirWS = GetCameraPositionWS() - IN.positionWS;

                half3 albedo = baseMap.xyz*_BaseColor;

                half3 diffuse = albedo * LightingLambert(light.color, light.direction, IN.normalWS);
                
                float3 halfVec = SafeNormalize(normalize(lightDirWS) + normalize(viewDirWS));
                half NdotH = saturate(dot(normalize(IN.normalWS), halfVec));
                half3 specular = light.color * specularColor.rgb * pow(NdotH, _Smoothness);

                float3 color = diffuse;// + specular;
                float shadow = MainLightRealtimeShadow(TransformWorldToShadowCoord(IN.positionWS));
                color *= shadow;

                // 自发光
                color += _EmitionCol * _EmitionStrength;


                
                // PBR，启动！
                // albedo
                float3 baseColor = GetAlbedo(IN.uv);
                // 粗糙度
                float3 roughness = GetRoughness(IN.uv);
                // 法线
                float3 normalTS = GetNormalTS(IN.uv);
                // 金属度
                float3 metallic = GetMetallic(IN.uv);

                // 变换到切线空间计算
                // 切线空间到世界空间的变换矩阵
                real3x3 T2W_MATRIX = CreateTangentToWorld(IN.normalWS, IN.tangentWS.xyz, IN.tangentWS.w);
                real3x3 W2T_MATRIX = Inverse(real4x4(
                                                        float4(T2W_MATRIX[0], 0),
                                                        float4(T2W_MATRIX[1], 0),
                                                        float4(T2W_MATRIX[2], 0),
                                                        float4(0,   0,   0,   1)));
                
                float3 lightDirTS = normalize(mul(light.direction, W2T_MATRIX));
                float3 viewDirTS = normalize(mul(IN.viewDirWS, W2T_MATRIX));
                float3 halfDirTS = normalize(lightDirTS + viewDirTS);
                

                // 计算圆锥体radiance
                // 7个60°圆锥体，两两相切组成
                // 先获取切线空间
                //切线空间采样方向（七个圆锥体的中心向量）
                float3 dirTS[7];
                dirTS[0] = normalize(float3( 0,  0      , 1));
                dirTS[1] = normalize(float3( 3,  1.73205, 2));
                dirTS[2] = normalize(float3( 0,  1.73205, 1));
                dirTS[3] = normalize(float3(-3,  1.73205, 2));
                dirTS[4] = normalize(float3(-3, -1.73205, 2));
                dirTS[5] = normalize(float3( 0, -1.73205, 1));
                dirTS[6] = normalize(float3( 3, -1.73205, 2));

                // 添加随机偏移
                float2 noiseUV0 = InterleavedGradientNoise3D(IN.positionWS * 4.6548792);
                float2 noiseUV1 = InterleavedGradientNoise3D(IN.positionWS * 7.911131719);
                float2 noiseUV2 = InterleavedGradientNoise3D(-2.7 * IN.positionWS * 2.3579);
                // 常数：1/根号3
                const float OneDividSqrtTree = 0.57735f * 1.55;
                float2 randDir = 2.0f * OneDividSqrtTree * float2(SAMPLE_TEXTURE2D(_WhiteNoise0, sampler_WhiteNoise0, noiseUV0).x, SAMPLE_TEXTURE2D(_WhiteNoise0, sampler_WhiteNoise0, noiseUV1).x) - 1.0 * OneDividSqrtTree;
                //randDir = 2.0 * ((noiseUV0 + noiseUV1 + noiseUV2) / 3.0) - 1.0;
                //randDir = 0;
                // dir[0]是法线，所以要特别计算
                dirTS[0] = normalize(dirTS[0] + float3(randDir.xy, 0));
                for (int i = 1; i < 7; ++i)
                {
                    float3 unitVecX = cross(dirTS[i], float3(0, 0, 1));
                    float3 unitVecY = cross(dirTS[i], unitVecX);
                    dirTS[i] = normalize(dirTS[i] + randDir.x * unitVecX + randDir.y * unitVecY);
                }
                
                // 将其变换为：世界空间
                float3 dir[7];
                dir[0] = mul(dirTS[0], T2W_MATRIX);
                dir[1] = mul(dirTS[1], T2W_MATRIX);
                dir[2] = mul(dirTS[2], T2W_MATRIX);
                dir[3] = mul(dirTS[3], T2W_MATRIX);
                dir[4] = mul(dirTS[4], T2W_MATRIX);
                dir[5] = mul(dirTS[5], T2W_MATRIX);
                dir[6] = mul(dirTS[6], T2W_MATRIX);
                
                //color += VoxelTexture[visIdxLODById(voxTexSize, 0, id)].col;
                float3 i_color = 0;
                // 沿着七个圆锥方向步进
                float3 marchPos[7];
                // 体素原点

                // 计算体素空间的包围盒
                float3 voxelAABBMaxPos = zeroPos + voxTexSize * voxSize;

                // debug
                float3 dirColors[7];
                float3 marchColors[7];
                for (int i = 0; i < 7; ++i)
                {
                    marchColors[i] = 0;
                }

                float sumWeight = 0;// 权重之和
                for (int i = 0; i < 7; ++i){
                    float3 cur_color = 0;
                    dirColors[i] = 0;
                    
                    marchPos[i] = IN.positionWS;// + 0.415 * voxSize * dir[i];    // 初始化
                    // 步进
                    // 初始步进距离：根号3 / 2 * 最小体素边长
                    // 每次步进都将步进距离翻倍
                    float marchDist = 1.73205 / 2.0 * voxSize;
                    // 每次步进，当前体素边长都翻倍
                    float currVoxSize = voxSize;
                    // 每次步进，当前体素体积都*8
                    int currVoxVol = 1;
                    // 权重步进动态维护
                    float weight = 1.0f;
                    // 每次步进，访问LOD级别都增大
                    for (int lodLevel = 0; lodLevel < _LodLevel; ++lodLevel)
                    {
                        if (weight <= 0.01) break;
                        
                        marchPos[i] += marchDist * dir[i];
                        // 将步进位置框在AABB内
                        marchPos[i].x = Clamp(marchPos[i].x, zeroPos.x, voxelAABBMaxPos.x);
                        marchPos[i].y = Clamp(marchPos[i].y, zeroPos.y, voxelAABBMaxPos.y);
                        marchPos[i].z = Clamp(marchPos[i].z, zeroPos.z, voxelAABBMaxPos.z);

                        uint3 curr_id = getId(GetCameraPositionWS(), marchPos[i], voxTexSize, voxSize);

                        // 采样
                        float3 tmp_col = 0;

                        // TODO：将采样变成正确的插值采样
                        /*linearSampleInfo sampleInfo = sampleVoxLinear(GetCameraPositionWS(), marchPos[i], voxTexSize, voxSize, lodLevel);
                        for (int j = 0; j < 8; ++j)
                        {
                            tmp_col += VoxelTexture[sampleInfo.visId[j]].col * sampleInfo.posWeight[j];
                            //tmp_voxvol += (float)VoxelTexture[sampleInfo.visId[j]].flags.x * sampleInfo.posWeight[j];
                        }*/

                        // 临时：现在不是插值采样
                        uint3 idxx = (marchPos[i] - zeroPos) / voxSize;
                        int curTexSize = voxTexSize / int(pow(2, lodLevel));
                        uint3 curID;
                        curID = idxx / int(pow(2, lodLevel));
                        int tmpSize = voxTexSize * voxTexSize * voxTexSize;
                        int visId = int(tmpSize * (8 - pow(0.125, lodLevel - 1))) / 7 + curID.x * curTexSize * curTexSize + curID.y * curTexSize + curID.z;
                        tmp_col = VoxelTexture[visId].col;
                        
                        // 采样该体素的法线
                        // 用voxNorm和marchingDir，计算一个权重
                        if (VoxelTexture[visId].flags.x != 0)
                        {
                            float angleWeight = (dot(dir[i], -normalize(VoxelTexture[visId].norm)) * 0.5 + 0.5);
                            
                            // PBR，启动
                            // 计算该光源到该像素的渲染BRDF
                            //float3 brdf0 = BRDF(lightDirTS, -dirTS[i], normalize(VoxelTexture[visId].norm), tmp_col, )
                            // 计算BRDF颜色
                            float3 brdf = BRDF(normalize(dirTS[i]),viewDirTS, normalTS, baseColor, metallic, roughness);
                            // 渲染方程
                            float3 Lo = tmp_col * max(dot(lightDirTS, normalTS), 0) * brdf;
                            
                            cur_color += Lo;// * angleWeight;// * currWeight;
                            float currWeight = 1;
                            sumWeight += currWeight;
                        }
                        
                        marchDist *= 2;
                        currVoxSize *= 2;
                        currVoxVol *= 8;
                    }
                    /*if (weight > 0.01)
                        cur_color /= 1.0f - saturate(weight);*/
                    i_color += cur_color;
                    dirColors[i] = cur_color;
                }
                //i_color /= sumWeight;

                // 魔法数，调一下强度
                //i_color *= 100;

                // 部分吸收
                color = i_color * albedo * 0.9 + i_color * 0.1;
                // 不吸收
                //color = i_color;

                // debug

                // 调试每一层步进采样的值
                /*float3 debugColor;
                debugColor = marchColors[6];// / 7.0f;
                color = debugColor * albedo * 0.9 + debugColor * 0.1;*/

                /*linearSampleInfo sampleInfo = sampleVoxLinear(GetCameraPositionWS(), IN.positionWS, voxTexSize, voxSize, 0);
                color = VoxelTexture[sampleInfo.visId[0]].col;*/
                
                //color = normalize(VoxelTexture[visIdxLODById(voxTexSize, 2, id)].norm * 0.5 + 0.5);
                //color = VoxelTexture[visIdxLODById(voxTexSize, 0, id)].norm * 0.5 + 0.5;
                
                //color = float3(0.1, 0.5, 0.6);

                //color = length(dir[0] - IN.normalWS);

                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
        UsePass "Universal Render Pipeline/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Lit/DepthNormals"
        UsePass "Universal Render Pipeline/Lit/ShadowCaster"    // 产生投影
    }
}

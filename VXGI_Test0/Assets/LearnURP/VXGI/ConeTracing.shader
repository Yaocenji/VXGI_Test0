Shader "LearnURP/ConeTracing"
{
    Properties
    {
        _BaseMap ("Base Texture",2D) = "white"{}
        _BaseColor("Base Color",Color)=(1,1,1,1)
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

            TEXTURE2D(_BaseMap); SAMPLER(sampler_BaseMap);
            TEXTURE2D(_WhiteNoise0); SAMPLER(sampler_WhiteNoise0);
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
                float shadow = MainLightRealtimeShadow(IN.shadowCoord);
                color *= shadow;

                // 自发光
                color += _EmitionCol * _EmitionStrength;

                // debug 
                //color = VoxelTexture[visId].norm * 0.5 + 0.5;
                //color = dir[0] * 0.5 + 0.5;

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

            TEXTURE2D(_BaseMap); SAMPLER(sampler_BaseMap);
            TEXTURE2D(_WhiteNoise0); SAMPLER(sampler_WhiteNoise0);
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
                OUT.shadowCoord = GetShadowCoord(positionInputs);
                return OUT;
            }

            float Clamp(float x, float l, float r)
            {
                if (x > r)
                    return r;
                if (x < l)
                    return l;
                return x;
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
                float shadow = MainLightRealtimeShadow(IN.shadowCoord);
                color *= shadow;

                // 自发光
                color += _EmitionCol * _EmitionStrength;

                // 计算圆锥体radiance
                // 7个60°圆锥体，两两相切组成

                // 先获取切线空间
                // 切线空间到世界空间的变换矩阵
                real3x3 T2W_MATRIX = CreateTangentToWorld(IN.normalWS, IN.tangentWS.xyz, IN.tangentWS.w);
                //切线空间采样方向（七个圆锥体的中心向量）
                float3 dir[7];
                dir[0] = float3( 0,  0      , 1);
                dir[1] = float3( 3,  1.73205, 2);
                dir[2] = float3( 0,  1.73205, 1);
                dir[3] = float3(-3,  1.73205, 2);
                dir[4] = float3(-3, -1.73205, 2);
                dir[5] = float3( 0, -1.73205, 1);
                dir[6] = float3( 3, -1.73205, 2);

                // 添加随机偏移
                float2 noiseUV0 = InterleavedGradientNoise3D(IN.positionWS * 4.6548792);
                float2 noiseUV1 = InterleavedGradientNoise3D(IN.positionWS * 7.911131719);
                float2 randDir = 4.0f * float2(SAMPLE_TEXTURE2D(_WhiteNoise0, sampler_WhiteNoise0, noiseUV0).x, SAMPLE_TEXTURE2D(_WhiteNoise0, sampler_WhiteNoise0, noiseUV1).x) - 2.0f;
                //randDir = 0;
                for (int i = 0; i < 7; ++i)
                    dir[i] = normalize(dir[i] + float3(randDir.xy, 0));
                
                // 将其变换为：世界空间
                dir[0] = mul(dir[0], T2W_MATRIX);
                dir[1] = mul(dir[1], T2W_MATRIX);
                dir[2] = mul(dir[2], T2W_MATRIX);
                dir[3] = mul(dir[3], T2W_MATRIX);
                dir[4] = mul(dir[4], T2W_MATRIX);
                dir[5] = mul(dir[5], T2W_MATRIX);
                dir[6] = mul(dir[6], T2W_MATRIX);
                
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

                        /*
                        // 计算当前体素的的中心点
                        uint3 curID = id / int(pow(2, i));
                        float3 voxCenterPos = zeroPos + curID * currVoxSize + 0.5f * currVoxSize;
                        
                        // 当前体素中心点投影到步进圆盘面的位置 与 步进中心点距离
                        float dist0 = normalize(cross(voxCenterPos - marchPos[i], dir[i]));
                        dist0 = Clamp(dist0, 0, currVoxSize);
                        
                        float h = sqrt(currVoxSize * currVoxSize - dist0 * dist0);
                        float tri_area = h * dist0 / 2.0;

                        float cosTheta = dist0 / currVoxSize;
                        float Theta = 4 * acos(cosTheta);
                        float sector_area = Theta * currVoxSize * currVoxSize / 8.0f;

                        // 圆面积
                        float circle_area = PI * currVoxSize * currVoxSize / 4.0f;
                        // 相交面积
                        float intersect_area = Clamp(sector_area - tri_area,
                                                        0, circle_area);
                        // 面积占比
                        float area_ratio = intersect_area / circle_area;
                        // 根据占比累加采样
                        float currWeight = weight * area_ratio;
                        */

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
                        float angleWeight = sqrt( saturate(dot(dir[i], -normalize(VoxelTexture[visId].norm)) + 0.15f) );    // sqrt和0.15f是模拟散射光波瓣的
                        
                        float currWeight = 1.0 / (_LodLevel - 1) * angleWeight;
                        sumWeight += currWeight;
                        //currWeight = weight * (tmp_voxvol / currVoxVol);
                        
                        //cur_color += VoxelTexture[visIdxLODById(voxTexSize, lodLevel, curr_id)].col / 6.0f;
                        
                        cur_color += tmp_col * currWeight;
                        marchColors[lodLevel] += tmp_col * currWeight;
                        //weight -= currWeight;
                        
                        marchDist *= 2;
                        currVoxSize *= 2;
                        currVoxVol *= 8;
                    }
                    /*if (weight > 0.01)
                        cur_color /= 1.0f - saturate(weight);*/
                    i_color += cur_color;
                    dirColors[i] = cur_color;
                }
                i_color /= sumWeight;

                // 魔法数，调一下颜色
                i_color *= .5;

                // 部分吸收
                color = i_color * albedo * 0.9 + i_color * 0.1;

                // debug

                // 调试每一层步进采样的值
                /*float3 debugColor;
                debugColor = marchColors[6];// / 7.0f;
                color = debugColor * albedo * 0.9 + debugColor * 0.1;*/

                /*linearSampleInfo sampleInfo = sampleVoxLinear(GetCameraPositionWS(), IN.positionWS, voxTexSize, voxSize, 0);
                color = VoxelTexture[sampleInfo.visId[0]].col;*/
                
                //color = normalize(VoxelTexture[visIdxLODById(voxTexSize, 2, id)].norm * 0.5 + 0.5);
                //color = VoxelTexture[visIdxLODById(voxTexSize, 0, id)].col;
                
                //color = float3(0.1, 0.5, 0.6);

                //color = IN.normalWS * 0.5 + 0.5;

                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
        UsePass "Universal Render Pipeline/Lit/DepthOnly"
        UsePass "Universal Render Pipeline/Lit/DepthNormals"
        UsePass "Universal Render Pipeline/Lit/ShadowCaster"    // 产生投影
    }
}

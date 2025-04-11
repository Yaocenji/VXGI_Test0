

struct VoxelData{
    uint3 flags;    // flags.x是否为0表示该体素是否为空；// TODO flags.y表示有多少个片元已经访问过该体素
    float3 col;     // 颜色
};

cbuffer UnityPerMaterial
{
};
float3 zeroPos;



// 根据世界空间位置，得到对应的最细体素三维坐标
uint3 getId(float3 cameraPos, float3 targetPos, int voxTexSize, float voxSize)
{
    //float3 zeroPos = cameraPos - (voxTexSize * voxSize / 2.0f).xxx;
    uint3 id = (targetPos - zeroPos) / voxSize;
    return id;
}

bool posLegal(uint3 id, int voxTexSize)
{
    return id.x < voxTexSize && id.y < voxTexSize && id.z < voxTexSize;
}

// 根据最细体素三维坐标，得到对应最细的一维读值坐标
int visitVoxIndex(uint3 id, int voxTexSize) {
    return id.x * voxTexSize * voxTexSize + id.y * voxTexSize + id.z;
}

// 根据世界空间位置，得到对应lod的一维读值坐标
int visIdxLOD(float3 cameraPos, float3 targetPos, int voxTexSize, float voxSize, int lodlevel)
{
    uint3 id = getId(cameraPos, targetPos, voxTexSize, voxSize);
    int curTexSize;
    uint3 curID;
    curTexSize = voxTexSize / int(pow(2, lodlevel));
    curID = id / int(pow(2, lodlevel));

    int tmpSize = voxTexSize * voxTexSize * voxTexSize;
    
    int visId;
    visId = int(tmpSize * (8 - pow(0.125, lodlevel - 1))) / 7;
    visId += curID.x * curTexSize * curTexSize + curID.y * curTexSize + curID.z;
    return visId;
}

// 根据最细体素三维坐标，得到对应lod的一维读值坐标
int visIdxLODById(int voxTexSize, int lodlevel, uint3 id)
{
    int curTexSize;
    uint3 curID;
    curTexSize = voxTexSize / int(pow(2, lodlevel));
    curID = id / int(pow(2, lodlevel));

    int tmpSize = voxTexSize * voxTexSize * voxTexSize;
    
    int visId;
    visId = int(tmpSize * (8 - pow(0.125, lodlevel - 1))) / 7;
    visId += curID.x * curTexSize * curTexSize + curID.y * curTexSize + curID.z;
    return visId;
}

struct linearSampleInfo
{
    int visId[8];
    float posWeight[8]; // 8个位置对应8个权重
};
// 根据世界空间坐标和给定LOD，返回三维空间采样
linearSampleInfo sampleVoxLinear(float3 cameraPos, float3 targetPos, int voxTexSize, float voxSize, int lodlevel)
{
    //float3 zeroPos = cameraPos - (voxTexSize * voxSize / 2.0f).xxx;
    uint3 id = (targetPos - zeroPos) / voxSize;
    
    int curTexSize = voxTexSize / int(pow(2, lodlevel));
    float currVoxSize = voxSize * pow(2, lodlevel);
    
    uint3 curID;
    curID = id / int(pow(2, lodlevel));

    // 体素中心点
    float3 voxCenterPos = zeroPos + curID * currVoxSize + 0.5f * currVoxSize;

    uint3 sampleID[8];
    sampleID[0] = uint3(curID.x - 1, curID.y - 1, curID.z - 1);
    sampleID[1] = uint3(curID.x    , curID.y - 1, curID.z - 1);
    sampleID[2] = uint3(curID.x - 1, curID.y - 1, curID.z    );
    sampleID[3] = uint3(curID.x    , curID.y - 1, curID.z    );
    
    sampleID[4] = uint3(curID.x - 1, curID.y    , curID.z - 1);
    sampleID[5] = uint3(curID.x    , curID.y    , curID.z - 1);
    sampleID[6] = uint3(curID.x - 1, curID.y    , curID.z    );
    sampleID[7] = uint3(curID.x    , curID.y    , curID.z    );

    float3 sampleZeroPos = zeroPos + sampleID[0] * currVoxSize + 0.5f * currVoxSize;

    if (targetPos.x > voxCenterPos.x)
    {
        sampleZeroPos.x += currVoxSize;
        for (int i = 0; i < 8; ++i)
            sampleID[i].x ++;
    }
    if (targetPos.y > voxCenterPos.x)
    {
        sampleZeroPos.y += currVoxSize;
        for (int i = 0; i < 8; ++i)
            sampleID[i].y ++;
    }
    if (targetPos.z > voxCenterPos.x)
    {
        sampleZeroPos.z += currVoxSize;
        for (int i = 0; i < 8; ++i)
            sampleID[i].z ++;
    }
    // 相对采样位置（01位置）
    float3 relativeSamplePos = (targetPos - sampleZeroPos) / currVoxSize;

    linearSampleInfo ans;
    
    int tmpSize = voxTexSize * voxTexSize * voxTexSize;
    int visId;
    for (int i = 0; i < 8; ++i)
    {
        visId = int(tmpSize * (8 - pow(0.125, lodlevel - 1))) / 7;
        visId += sampleID[i].x * curTexSize * curTexSize + sampleID[i].y * curTexSize + sampleID[i].z;
        ans.visId[i] = visId;
    }
    ans.posWeight[0] = (1 - relativeSamplePos.x) * (1 - relativeSamplePos.y) * (1 - relativeSamplePos.z);
    ans.posWeight[1] = relativeSamplePos.x * (1 - relativeSamplePos.y) * (1 - relativeSamplePos.z);
    ans.posWeight[2] = (1 - relativeSamplePos.x) * (1 - relativeSamplePos.y) * relativeSamplePos.z;
    ans.posWeight[3] = relativeSamplePos.x * (1 - relativeSamplePos.y) * relativeSamplePos.z;

    ans.posWeight[4] = (1 - relativeSamplePos.x) * relativeSamplePos.y * (1 - relativeSamplePos.z);
    ans.posWeight[5] = relativeSamplePos.x * relativeSamplePos.y * (1 - relativeSamplePos.z);
    ans.posWeight[6] = (1 - relativeSamplePos.x) * relativeSamplePos.y * relativeSamplePos.z;
    ans.posWeight[7] = relativeSamplePos.x * relativeSamplePos.y * relativeSamplePos.z;
    return ans;
}

// 根据一维ID获取三维坐标
#define VIS_VOX_IDX(id)         \
    int visId = id.x * voxTexSize * voxTexSize + id.y * voxTexSize + id.z;

// 根据一维ID获取三维坐标
#define VIS_VOX_IDX_LOD(id, lod)                                            \
    int curTexSize = voxTexSize;                                            \
    for (int i = 0; i < lod; ++i)                                           \
        curTexSize /= 2;                                                    \
    int visId = id.x * curTexSize * curTexSize + id.y * curTexSize + id.z;

/*#define GET_ID(cameraPos, targetPos, voxTexSize, voxSize)           \
    float3 zeroPos = cameraPos - (voxTexSize * voxSize / 2.0f).xxx; \
    uint3 id = ((targetPos - zeroPos) / voxSize).xxx;*/
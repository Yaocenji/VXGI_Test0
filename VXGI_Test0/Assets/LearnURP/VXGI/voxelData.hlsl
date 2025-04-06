

struct VoxelData{
    uint3 flags;    // flags.x是否为0表示该体素是否为空；// TODO flags.y表示有多少个片元已经访问过该体素
    float3 col;     // 颜色
};

int visitVoxIndex(uint3 id, int voxTexSize) {
    return id.x * voxTexSize * voxTexSize + id.y * voxTexSize + id.z;
}

uint3 getId(float3 cameraPos, float3 targetPos, int voxTexSize, float voxSize)
{
    float3 zeroPos = cameraPos - (voxTexSize * voxSize / 2.0f).xxx;
    uint3 id = (targetPos - zeroPos) / voxSize;
    return id;
}

bool posLegal(uint3 id, int voxTexSize)
{
    return id.x < voxTexSize && id.y < voxTexSize && id.z < voxTexSize;
}

// 根据lod读取数据

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

// 根据一维ID获取三维坐标
#define VIS_VOX_IDX(id)         \
    int visId = id.x * voxTexSize * voxTexSize + id.y * voxTexSize + id.z;

// 根据一维ID获取三维坐标
#define VIS_VOX_IDX_LOD(id, lod)                                            \
    int curTexSize = voxTexSize;                                            \
    for (int i = 0; i < lod; ++i)                                           \
        curTexSize /= 2;                                                    \
    int visId = id.x * curTexSize * curTexSize + id.y * curTexSize + id.z;

#define GET_ID(cameraPos, targetPos, voxTexSize, voxSize)           \
    float3 zeroPos = cameraPos - (voxTexSize * voxSize / 2.0f).xxx; \
    uint3 id = ((targetPos - zeroPos) / voxSize).xxx;
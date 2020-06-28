# pandas version
def Coordinate2MeshCodePandas(dLng, dLat):
    # cf: http://white-bear.info/archives/1400
    # Make sure the input values are decimal
    iMeshCode_1stMesh_Part_p = dLat *60 // 40
    iMeshCode_1stMesh_Part_u = ( dLng - 100 ) // 1
    iMeshCode_2ndMesh_Part_q = dLat *60 % 40 // 5
    iMeshCode_2ndMesh_Part_v = ( ( dLng - 100 ) % 1 ) * 60 // 7.5
    iMeshCode_3rdMesh_Part_r = dLat *60 % 40 % 5 * 60 // 30
    iMeshCode_3rdMesh_Part_w = ( ( dLng - 100 ) % 1 ) * 60 % 7.5 * 60 // 45
    iMeshCode = iMeshCode_1stMesh_Part_p * 1000000 + iMeshCode_1stMesh_Part_u * 10000 + \
                iMeshCode_2ndMesh_Part_q * 1000 + \
                iMeshCode_2ndMesh_Part_v * 100 + \
                iMeshCode_3rdMesh_Part_r * 10 + \
                iMeshCode_3rdMesh_Part_w
    return iMeshCode.astype(int).astype(str)


def Coordinate2MeshCode(dLng, dLat):
    # cf: http://white-bear.info/archives/1400
    # Make sure the input values are decimal
    iMeshCode_1stMesh_Part_p = dLat *60 // 40
    iMeshCode_1stMesh_Part_u = ( dLng - 100 ) // 1
    iMeshCode_2ndMesh_Part_q = dLat *60 % 40 // 5
    iMeshCode_2ndMesh_Part_v = ( ( dLng - 100 ) % 1 ) * 60 // 7.5
    iMeshCode_3rdMesh_Part_r = dLat *60 % 40 % 5 * 60 // 30
    iMeshCode_3rdMesh_Part_w = ( ( dLng - 100 ) % 1 ) * 60 % 7.5 * 60 // 45
    iMeshCode = iMeshCode_1stMesh_Part_p * 1000000 + iMeshCode_1stMesh_Part_u * 10000 + \
                iMeshCode_2ndMesh_Part_q * 1000 + \
                iMeshCode_2ndMesh_Part_v * 100 + \
                iMeshCode_3rdMesh_Part_r * 10 + \
                iMeshCode_3rdMesh_Part_w
    return str(int(iMeshCode))


def parse_MeshCode(mesh_code):

   # convert mesh no to coordinate
   # :return: lat and lon

    LAT_HEIGHT_MESH1 = 0.6666
    LAT_HEIGHT_MESH2 = 0.0833
    LNG_WIDTH_MESH2 = 0.125
    LAT_HEIGHT_MESH3 = 0.0083
    LNG_WIDTH_MESH3 = 0.0125
    LAT_HEIGHT_MESH4 = 0.004166
    LNG_WIDTH_MESH4 = 0.00625
    LAT_HEIGHT_MESH5 = 0.002083
    LNG_WIDTH_MESH5 = 0.003125
    strlen = len(mesh_code)
    if strlen == 0 or strlen > 11:
        return None
    x = 0.000000001
    y = 0.000000001
    if strlen >= 4:
        y += float(LAT_HEIGHT_MESH1*int(mesh_code[0: 2]))
        x += 100 + int(mesh_code[2: 4])

    if strlen >= 6:
         y += float(LAT_HEIGHT_MESH2*int(mesh_code[4: 5]))
         x += float(LNG_WIDTH_MESH2*int(mesh_code[5: 6]))

    if strlen >= 8:
        y += float(LAT_HEIGHT_MESH3 * int(mesh_code[6: 7]))
        x += float(LNG_WIDTH_MESH3 * int(mesh_code[7: 8]))

    if strlen >= 9:
        n = int(mesh_code[8: 9])
        y += float(LAT_HEIGHT_MESH4*(0 if n <= 2 else 1))
        x += float(LNG_WIDTH_MESH4*(0 if n % 2 == 1 else 1))

    if strlen >= 10:
        n = int(mesh_code[9: 10])
        y += float(LAT_HEIGHT_MESH5 * (0 if n <= 2 else 1))
        x += float(LNG_WIDTH_MESH5 * (0 if n % 2 == 1 else 1))

    return x, y
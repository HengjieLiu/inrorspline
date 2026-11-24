import numpy as np
from scripts.visualization.lungct_viewer.geometry import PhysicalField, compact, display_geometry, invert_points, to_lps


def test_lps_orientation_and_round_trip():
    sp=np.array([.625,.625,2.5]);p=np.array([[2,3,4],[8,9,10]],float)
    lps=to_lps(p,sp,121)
    assert lps[1,0]>lps[0,0] and lps[1,1]>lps[0,1] and lps[1,2]<lps[0,2]
    restored=lps.copy();restored[:,2]=(121-1)*sp[2]-lps[:,2]
    np.testing.assert_array_equal(restored/sp,p)


def test_endpoint_spacing_not_nominal_one_mm():
    a=dict(registration_shape_zyx=[272,201,229],crop_shape_zyx=[109,321,366],
           crop_bounds_zyx_inclusive=[[0,108],[114,434],[70,435]],native_spacing_zyx_mm=[2.5,.625,.625],raw_shape_zyx=[121,512,512])
    g=display_geometry(a)
    np.testing.assert_allclose(g['step_lps_mm'],[365*.625/228,320*.625/200,-108*2.5/271])
    assert g['shape_zyx'][0]==272 and g['step_lps_mm'][2]!=-1


def test_pull_translation_inverse_and_axis_sign():
    shape=(21,23,25);data=np.zeros((*shape,3),np.float32);data[:]=[1,-2,3]
    g=dict(crop_bounds_zyx_inclusive=[[0,20],[0,22],[0,24]],crop_shape_zyx=list(shape),registration_shape_zyx=list(shape))
    field=PhysicalField(data,[1,1,1],[[0,24],[0,22],[0,20]],g)
    p=np.array([[12,11,10],[16,14,8]],float)
    np.testing.assert_array_equal(field.sample(p),[[3,-2,1],[3,-2,1]])
    inv=invert_points(field,p)
    assert all(v['status']=='resolved' for v in inv)
    np.testing.assert_allclose([v['point_native_xyz'] for v in inv],p-[3,-2,1],atol=.005)
    assert to_lps([0,0,1],[1,1,1],21)[2]-to_lps([0,0,0],[1,1,1],21)[2]==-1


def test_zero_inverse_and_invalid_point():
    field=PhysicalField(np.zeros((3,9,9,9),np.float32),[1,1,1],[[0,8]]*3,ptv=True)
    r=invert_points(field,[[3,4,5],[100,100,100]])
    np.testing.assert_allclose(r[0]['point_native_xyz'],[3,4,5])
    assert r[1]['status']=='unresolved'
    assert not field.valid([[9,2,2]])[0]


def test_compact_precision_and_fallback():
    a=np.array([0,1.2345,-1024,3000],np.float32)
    c,e=compact(a,2);assert c.dtype==np.dtype('<f2') and e<=2
    c,e=compact(np.array([123456.789],np.float32),.1)
    assert c.dtype==np.dtype('<f4') and e==0


def test_nonzero_crop_field_geometry():
    data=np.zeros((9,9,9,3),np.float32)
    data[...,2]=np.arange(9)[None,None,:]
    g=dict(crop_bounds_zyx_inclusive=[[3,7],[4,8],[5,9]],crop_shape_zyx=[5]*3,registration_shape_zyx=[9]*3)
    field=PhysicalField(data,[1]*3,[[5,9],[4,8],[3,7]],g)
    np.testing.assert_allclose(field.sample([[6,5,4]]),[[2,0,0]])


def test_translation_warp_mask_and_difference():
    from scipy.ndimage import map_coordinates
    image=np.arange(7,dtype=np.float32)[None,None,:]+np.zeros((5,6,7),np.float32)
    mask=(image>3).astype(np.uint8)
    z,y,x=np.meshgrid(np.arange(5),np.arange(6),np.arange(7),indexing='ij')
    warped=map_coordinates(image,[z,y,x+1],order=1,mode='constant',cval=0)
    labels=map_coordinates(mask,[z,y,x+1],order=0,mode='constant',cval=0)
    np.testing.assert_array_equal(warped[:,:,:-1]-image[:,:,:-1],1)
    np.testing.assert_array_equal(labels,warped>3)


def test_inverse_marks_multiple_roots():
    field=np.zeros((3,11,11,11),np.float32)
    x=np.arange(11,dtype=np.float32)
    field[0]=((x-5)**2/5-x)[None,None,:]
    f=PhysicalField(field,[1]*3,[[0,10]]*3,ptv=True)
    result=invert_points(f,[[1.8,5,5]])[0]
    assert result['status']=='ambiguous' and result['roots']>=2
    assert result['inverse_residual_mm']<=.05


def test_readonly_http_routes_and_gzip(tmp_path):
    import gzip,json,http.client,threading
    from pathlib import Path
    from scripts.visualization.lungct_viewer.serve import Server,Handler
    (tmp_path/'catalog.json').write_text('{"cases":[]}')
    case=tmp_path/'cases/4dct_01';case.mkdir(parents=True)
    (case/'image.bin.gz').write_bytes(gzip.compress(b'1234'))
    (case/'manifest.json').write_text(json.dumps(dict(id='4dct_01',source_hashes={'private':'secret'})))
    server=Server(('127.0.0.1',0),Handler);server.bundle_root=tmp_path;server.web_root=Path(__file__).parents[1]/'scripts/visualization/lungct_viewer/web'
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    conn=http.client.HTTPConnection(*server.server_address)
    try:
        conn.request('GET','/assets/cases/4dct_01/manifest.json');r=conn.getresponse();assert r.status==200
        assert 'source_hashes' not in json.loads(r.read())
        conn.request('GET','/assets/cases/4dct_01/image.bin.gz');r=conn.getresponse()
        assert r.getheader('Content-Encoding')=='gzip' and gzip.decompress(r.read())==b'1234'
        for path in ['/assets/catalog.json','/assets/preparation_123.json','/vendor/../catalog.json','/../AGENTS.md']:
            conn.request('GET',path);r=conn.getresponse();assert r.status==404;r.read()
        conn.request('POST','/');r=conn.getresponse();assert r.status==501;r.read()
    finally:conn.close();server.shutdown();server.server_close();thread.join()

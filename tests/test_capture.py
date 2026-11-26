import json
import numpy as np
import pytest
from PIL import Image
from perception.capture.prepare import prepare
from perception.capture.segment import segment


def photos(tmp_path, count=3):
    source = tmp_path / 'photos'
    source.mkdir()
    for i in range(count):
        Image.new('RGB', (80, 60), (i * 60, 80, 120)).save(source / f'photo{i+1}.png')
    return source


def test_prepare_order_resize_and_manifest(tmp_path):
    source = photos(tmp_path)
    Image.new('RGB', (80, 60), 'white').save(source / 'photo10.png')
    result = prepare(source, tmp_path / 'scene', max_side=40)
    assert [f['source_name'] for f in result['frames']] == ['photo1.png', 'photo2.png', 'photo3.png', 'photo10.png']
    assert all(f['size'] == [40, 30] for f in result['frames'])
    assert result['geometry']['scale_meters'] is None
    assert (tmp_path / 'scene/contact-sheet.jpg').exists()
    with pytest.raises(ValueError, match='already exists'):
        prepare(source, tmp_path / 'scene')


def test_reject_bad_frames_and_cleanup(tmp_path):
    source = photos(tmp_path)
    (source / 'photo2.png').write_text('not an image')
    with pytest.raises(OSError):
        prepare(source, tmp_path / 'scene')
    assert not (tmp_path / 'scene').exists()
    assert not list(tmp_path.glob('.capture-*'))
    with pytest.raises(ValueError):
        prepare(source, source / 'scene')
    with pytest.raises(ValueError):
        prepare(source, tmp_path / 'scene', sample_fps=float('nan'))


def predictions(count):
    for index in range(count):
        mask = np.zeros((60, 80), bool)
        mask[10:30, 20+index:40+index] = True
        yield index, {'out_obj_ids': np.array([7]), 'out_binary_masks': np.array([mask]), 'out_probs': np.array([.9])}


def test_track_export_and_mask_alignment(tmp_path):
    scene = tmp_path / 'scene'
    prepare(photos(tmp_path), scene)
    out = tmp_path / 'tracks'
    result = segment(scene, out, 'cup', outputs=predictions(3))
    assert result['backend'] == 'external_test_outputs'
    assert len(result['objects']) == 1
    track = result['objects'][0]
    assert len(track['observations']) == 3
    assert track['physical_properties']['status'] == 'unknown'
    for record in track['observations']:
        with Image.open(out / record['mask']) as mask:
            assert mask.size == (80, 60)
            assert mask.getbbox() == tuple(record['bbox_xyxy'])
        with Image.open(out / record['crop']) as crop:
            assert crop.mode == 'RGBA' and crop.size == (20, 20)


def test_empty_detections_are_valid(tmp_path):
    scene = tmp_path / 'scene'
    prepare(photos(tmp_path, 1), scene)
    result = segment(scene, tmp_path / 'tracks', 'cup', outputs=[(0, {
        'out_obj_ids': [], 'out_binary_masks': [], 'out_probs': []})])
    assert result['objects'] == []


def test_missing_frames_and_modified_input_fail(tmp_path):
    scene = tmp_path / 'scene'
    prepare(photos(tmp_path), scene)
    with pytest.raises(ValueError, match='every prepared frame'):
        segment(scene, tmp_path / 'tracks', 'cup', outputs=predictions(2))
    assert not (tmp_path / 'tracks').exists()
    (scene / 'images/000000.jpg').write_bytes(b'changed')
    with pytest.raises(ValueError, match='manifest'):
        segment(scene, tmp_path / 'tracks', 'cup', outputs=predictions(3))


def test_video_sampling(tmp_path):
    cv2 = pytest.importorskip('cv2')
    source = tmp_path / 'clip.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (80, 60))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((60, 80, 3), i * 10, np.uint8))
    writer.release()
    result = prepare(source, tmp_path / 'scene', sample_fps=5, max_frames=3)
    assert len(result['frames']) == 3
    assert result['frames'][0]['timestamp_seconds'] == 0
    assert result['frames'][-1]['timestamp_seconds'] == pytest.approx(1.8)

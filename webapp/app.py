#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask dashboard for the Adaptive-Ensemble-Backdoor-Attack-Detection project.

Run with:
    venv\\Scripts\\python.exe webapp\\app.py
then open http://127.0.0.1:5050 in a browser.

Endpoints:
    GET  /                      dashboard page
    GET  /api/datasets          list .h5 files in data/
    GET  /api/models            list .h5 files in models/
    POST /api/upload            upload a .h5 dataset -> data/
    POST /api/poison            create a static poisoned copy of a dataset
    POST /api/train             train a model (clean or on-the-fly poisoned)
    POST /api/detect            run trigger reverse-engineering + MAD outlier scan
    GET  /api/status            current background job status/log/progress
    GET  /results/<path>        serve generated pattern/mask/fusion images
"""

import os
import sys
import time
import uuid
import threading
import traceback

from flask import Flask, request, jsonify, send_from_directory, send_file

sys.path.insert(0, os.path.dirname(__file__))
import pipeline  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------- #
# Single global job slot. This project's TF1/Keras stack is not safe to run
# multiple training/detection jobs concurrently in one process, so the
# dashboard runs one job at a time and reports its state via /api/status.
# ---------------------------------------------------------------------- #

JOB_LOCK = threading.Lock()
JOB = {
    'running': False,
    'kind': None,        # 'train' | 'poison' | 'detect'
    'started_at': None,
    'logs': [],
    'progress': {'done': 0, 'total': 0, 'stage': ''},
    'result': None,
    'error': None,
}


def _reset_job(kind):
    JOB['running'] = True
    JOB['kind'] = kind
    JOB['started_at'] = time.time()
    JOB['logs'] = []
    JOB['progress'] = {'done': 0, 'total': 0, 'stage': ''}
    JOB['result'] = None
    JOB['error'] = None


def _log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    JOB['logs'].append(line)
    print(line)


def _progress(done, total, stage=''):
    JOB['progress'] = {'done': done, 'total': total, 'stage': stage}


def _run_in_background(kind, target, kwargs):
    if JOB['running']:
        return False, 'A job is already running (%s). Wait for it to finish.' % JOB['kind']

    def _worker():
        try:
            result = target(log=_log, progress=_progress, **kwargs)
            JOB['result'] = result
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            JOB['error'] = str(exc)
            _log('ERROR: %s' % exc)
        finally:
            JOB['running'] = False

    with JOB_LOCK:
        _reset_job(kind)
        t = threading.Thread(target=_worker)
        t.daemon = True
        t.start()
    return True, None


# ---------------------------------------------------------------------- #
# Static / page routes
# ---------------------------------------------------------------------- #

@app.route('/')
def index():
    return send_file(os.path.join(APP_DIR, 'static', 'index.html'))


@app.route('/results/<path:path>')
def results(path):
    return send_from_directory(pipeline.RESULTS_DIR, path)


# ---------------------------------------------------------------------- #
# Dataset / model listing
# ---------------------------------------------------------------------- #

def _list_h5(folder):
    if not os.path.exists(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith('.h5'))


@app.route('/api/datasets')
def api_datasets():
    files = _list_h5(pipeline.DATA_DIR)
    out = []
    for f in files:
        path = os.path.join(pipeline.DATA_DIR, f)
        try:
            info = pipeline.dataset_info(path)
        except Exception as exc:  # noqa: BLE001
            info = {'error': str(exc)}
        info['name'] = f
        out.append(info)
    return jsonify(out)


@app.route('/api/models')
def api_models():
    return jsonify(_list_h5(pipeline.MODEL_DIR))


# ---------------------------------------------------------------------- #
# Upload
# ---------------------------------------------------------------------- #

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file field named "file"'}), 400
    f = request.files['file']
    if not f.filename.endswith('.h5'):
        return jsonify({'error': 'expected a .h5 file with keys X_train/Y_train/X_test/Y_test '
                                  '(or at least X_test/Y_test for detection-only use)'}), 400

    safe_name = os.path.basename(f.filename)
    dest = os.path.join(pipeline.DATA_DIR, safe_name)
    f.save(dest)

    try:
        info = pipeline.dataset_info(dest)
    except Exception as exc:  # noqa: BLE001
        os.remove(dest)
        return jsonify({'error': 'uploaded file is not a readable .h5 dataset: %s' % exc}), 400

    info['name'] = safe_name
    return jsonify(info)


# ---------------------------------------------------------------------- #
# Poison (produce a static poisoned dataset file)
# ---------------------------------------------------------------------- #

@app.route('/api/poison', methods=['POST'])
def api_poison():
    body = request.get_json(force=True)
    dataset = body.get('dataset')
    target_labels = body.get('target_labels', [0])
    pattern_size = int(body.get('pattern_size', 4))
    margin = int(body.get('margin', 1))
    poison_ratio = float(body.get('poison_ratio', 0.2))
    out_name = body.get('out_name') or ('poisoned_%s' % dataset)

    if not dataset:
        return jsonify({'error': 'dataset is required'}), 400

    src_path = os.path.join(pipeline.DATA_DIR, dataset)
    if not os.path.exists(src_path):
        return jsonify({'error': 'dataset not found: %s' % dataset}), 404
    dst_path = os.path.join(pipeline.DATA_DIR, out_name)

    ok, err = _run_in_background('poison', _poison_job, {
        'src_path': src_path, 'dst_path': dst_path,
        'target_labels': [int(t) for t in target_labels],
        'pattern_size': pattern_size, 'margin': margin,
        'poison_ratio': poison_ratio,
    })
    if not ok:
        return jsonify({'error': err}), 409
    return jsonify({'started': True, 'out_name': out_name})


def _poison_job(src_path, dst_path, target_labels, pattern_size, margin,
                 poison_ratio, log, progress):
    progress(0, 1, 'poisoning')
    path = pipeline.poison_dataset_to_file(
        src_path, dst_path, target_labels, pattern_size, margin, poison_ratio, log=log)
    progress(1, 1, 'poisoning')
    return {'dataset_path': path, 'name': os.path.basename(path)}


# ---------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------- #

@app.route('/api/train', methods=['POST'])
def api_train():
    body = request.get_json(force=True)
    dataset = body.get('dataset')
    mode = body.get('mode', 'clean')  # 'clean' | 'poison'
    model_name = body.get('model_name') or ('model_%s.h5' % uuid.uuid4().hex[:8])
    if not model_name.endswith('.h5'):
        model_name += '.h5'
    target_labels = body.get('target_labels', [0])
    pattern_size = int(body.get('pattern_size', 4))
    margin = int(body.get('margin', 1))
    inject_ratio = float(body.get('inject_ratio', 0.2))
    epochs = int(body.get('epochs', 10))

    if not dataset:
        return jsonify({'error': 'dataset is required'}), 400
    dataset_path = os.path.join(pipeline.DATA_DIR, dataset)
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'dataset not found: %s' % dataset}), 404

    ok, err = _run_in_background('train', pipeline.train_model, {
        'dataset_path': dataset_path, 'model_name': model_name, 'mode': mode,
        'target_labels': [int(t) for t in target_labels],
        'pattern_size': pattern_size, 'margin': margin,
        'inject_ratio': inject_ratio, 'epochs': epochs,
    })
    if not ok:
        return jsonify({'error': err}), 409
    return jsonify({'started': True, 'model_name': model_name})


# ---------------------------------------------------------------------- #
# Detect
# ---------------------------------------------------------------------- #

@app.route('/api/detect', methods=['POST'])
def api_detect():
    body = request.get_json(force=True)
    dataset = body.get('dataset')
    model = body.get('model')
    steps = int(body.get('steps', 200))
    batch_size = int(body.get('batch_size', 32))
    labels_to_scan = body.get('labels_to_scan')  # optional list of ints
    true_backdoor_labels = body.get('true_backdoor_labels')  # optional ground-truth list of ints

    if not dataset or not model:
        return jsonify({'error': 'dataset and model are required'}), 400

    dataset_path = os.path.join(pipeline.DATA_DIR, dataset)
    model_path = os.path.join(pipeline.MODEL_DIR, model)
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'dataset not found: %s' % dataset}), 404
    if not os.path.exists(model_path):
        return jsonify({'error': 'model not found: %s' % model}), 404

    run_id = time.strftime('run_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:6]

    ok, err = _run_in_background('detect', pipeline.run_detection, {
        'model_path': model_path, 'dataset_path': dataset_path, 'run_id': run_id,
        'steps': steps, 'batch_size': batch_size,
        'labels_to_scan': [int(t) for t in labels_to_scan] if labels_to_scan else None,
        'true_backdoor_labels': [int(t) for t in true_backdoor_labels] if true_backdoor_labels else None,
    })
    if not ok:
        return jsonify({'error': err}), 409
    return jsonify({'started': True, 'run_id': run_id})


# ---------------------------------------------------------------------- #
# Status polling
# ---------------------------------------------------------------------- #

@app.route('/api/status')
def api_status():
    return jsonify({
        'running': JOB['running'],
        'kind': JOB['kind'],
        'logs': JOB['logs'][-500:],
        'progress': JOB['progress'],
        'result': JOB['result'],
        'error': JOB['error'],
    })


if __name__ == '__main__':
    print('Project root:', PROJECT_ROOT)
    print('Data dir:    ', pipeline.DATA_DIR)
    print('Models dir:  ', pipeline.MODEL_DIR)
    print('Results dir: ', pipeline.RESULTS_DIR)
    app.run(host='127.0.0.1', port=5050, threaded=True, debug=False)

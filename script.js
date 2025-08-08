document.addEventListener('DOMContentLoaded', () => {
    // --- Global State & Elements ---
    const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);
    const dynamicContent = document.getElementById('dynamic-content');
    const videoElement = document.getElementById('camera-feed');
    const statusMessage = document.getElementById('status-message');
    const predictionOutput = document.getElementById('prediction-output');
    const cameraControlsContainer = document.getElementById('camera-controls-container');

    const trainNewModelBtn = document.getElementById('train-new-model-btn');
    const existingModelsBtn = document.getElementById('existing-models-btn');
    const modelCountSpan = document.getElementById('model-count');

    let localStream = null;
    let frameSender = null;

    // --- WebSocket Handlers ---
    socket.on('connect', () => console.log('WS: Connected'));
    socket.on('disconnect', () => console.log('WS: Disconnected'));
    socket.on('update_status', (data) => {
        console.log('WS: Status Update:', data.message);
        statusMessage.innerText = data.message;
    });
    socket.on('prediction', (data) => {
        console.log('WS: Prediction:', data.gesture, data.confidence);
        predictionOutput.innerText = `Prediction: ${data.gesture} ${data.confidence}`;
    });
    socket.on('confirm_training', (data) => {
        console.log('WS: Confirm Training:', data.message);
        showTrainingConfirmation(data.model_name, data.is_retraining, data.captured_data_count);
    });
    socket.on('training_complete', (data) => {
        console.log('WS: Training Complete:', data.message);
        alert(data.message);
        resetToWelcomeState();
    });
    socket.on('model_deleted', (data) => {
        console.log('WS: Model Deleted:', data.message);
        alert(data.message);
        resetToWelcomeState();
    });

    // --- Camera Management ---
    async function getCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'videoinput');
        } catch (err) {
            console.error("Error enumerating devices:", err);
            return [];
        }
    }

    async function startCamera(deviceId = null) {
        if (localStream) {
            stopCamera();
        }
        try {
            const constraints = { video: { deviceId: deviceId ? { exact: deviceId } : undefined } };
            localStream = await navigator.mediaDevices.getUserMedia(constraints);
            videoElement.srcObject = localStream;
            statusMessage.innerText = 'Camera ON';
            startFrameSending();
        } catch (err) {
            console.error("Error starting camera:", err);
            statusMessage.innerText = 'Error: Could not access camera.';
        }
    }

    function stopCamera() {
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            localStream = null;
        }
        if (frameSender) {
            clearInterval(frameSender);
            frameSender = null;
        }
        videoElement.srcObject = null;
        statusMessage.innerText = 'Camera OFF';
    }

    function startFrameSending() {
        if (frameSender) {
            clearInterval(frameSender);
        }
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        frameSender = setInterval(() => {
            if (localStream && !videoElement.paused && !videoElement.ended) {
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                socket.emit('stream_frame', { image: imageData });
            }
        }, 100); // 10 FPS
    }

    async function setupCameraControls() {
        const cameras = await getCameras();
        let controlsHtml = '';
        if (cameras.length > 1) {
            const options = cameras.map(cam => `<option value="${cam.deviceId}">${cam.label || `Camera ${cam.deviceId}`}</option>`).join('');
            controlsHtml += `<select id="camera-select">${options}</select>`;
        }
        controlsHtml += '<button id="close-camera-btn">Close Camera</button>';
        cameraControlsContainer.innerHTML = controlsHtml;

        const cameraSelect = document.getElementById('camera-select');
        if (cameraSelect) {
            cameraSelect.addEventListener('change', (e) => startCamera(e.target.value));
        }
        document.getElementById('close-camera-btn').addEventListener('click', stopCamera);

        // Start with the first camera by default
        if (cameras.length > 0) {
            startCamera(cameras[0].deviceId);
        }
    }

    // --- Core UI Functions ---
    async function resetToWelcomeState() {
        console.log('UI: Resetting to welcome state...');
        socket.emit('stop_test');
        predictionOutput.innerText = '';
        await updateModelCount();
        loadWelcomeUI();
    }

    function setActiveMenu(selectedBtn) {
        trainNewModelBtn.classList.remove('active');
        existingModelsBtn.classList.remove('active');
        if (selectedBtn) {
            selectedBtn.classList.add('active');
        }
    }

    async function updateModelCount() {
        try {
            const response = await fetch('/api/model_count');
            const count = await response.json();
            modelCountSpan.innerText = count;
        } catch (error) {
            console.error('Error fetching model count:', error);
            modelCountSpan.innerText = '-';
        }
    }

    // --- Event Listeners ---
    trainNewModelBtn.addEventListener('click', () => {
        setActiveMenu(trainNewModelBtn);
        loadNewModelUI();
    });

    existingModelsBtn.addEventListener('click', () => {
        setActiveMenu(existingModelsBtn);
        loadExistingModelsUI();
    });

    dynamicContent.addEventListener('click', async (e) => {
        const target = e.target;

        if (target.matches('#start-capture-btn')) {
            startNewCapture();
        } else if (target.matches('#confirm-train-btn')) {
            const modelName = target.dataset.modelName;
            const isRetraining = target.dataset.isRetraining === 'true';
            const capturedDataCount = parseInt(target.dataset.capturedDataCount);
            socket.emit('start_training', { model_name: modelName, is_retraining: isRetraining, captured_data_count: capturedDataCount });
            statusMessage.innerText = 'Training model...';
        } else if (target.matches('#retake-data-btn')) {
            const modelName = target.dataset.modelName;
            socket.emit('retake_data', { model_name: modelName });
        } else if (target.matches('.retrain-btn')) {
            loadRetrainOptionsUI(target.dataset.model);
        } else if (target.matches('.test-btn')) {
            startTest(target.dataset.model);
        } else if (target.matches('.delete-btn')) {
            if (confirm(`Are you sure you want to delete model '${target.dataset.model}'?`)) {
                socket.emit('delete_model', { model_name: target.dataset.model });
            }
        } else if (target.matches('#retrain-same-gestures-btn')) {
            const modelName = target.dataset.modelName;
            const captureTime = document.getElementById('capture-time-select-retrain').value;
            socket.emit('retrain_model', { model_name: modelName, retrain_option: 'same_gestures', capture_time: captureTime });
        } else if (target.matches('#retrain-new-gestures-btn')) {
            socket.emit('retrain_model', { model_name: target.dataset.modelName, retrain_option: 'new_gestures' });
        } else if (target.matches('#stop-test-btn')) {
            resetToWelcomeState();
        }
    });

    // --- UI Loaders ---
    function loadWelcomeUI() {
        dynamicContent.innerHTML = '<h2>Welcome!</h2><p>Select an option to begin.</p>';
        setActiveMenu(null);
    }

    function loadNewModelUI() {
        dynamicContent.innerHTML = `
            <div class="content-section">
                <h2>Create New Model</h2>
                <input type="text" id="model-name-input" placeholder="Enter a name for your new model...">
                <div class="capture-time-container">
                    <label for="capture-time-select">Capture Time (seconds):</label>
                    <select id="capture-time-select">
                        <option value="10">10</option>
                        <option value="15">15</option>
                        <option value="20">20</option>
                    </select>
                </div>
                <div id="gesture-tags-container"></div>
                <button id="start-capture-btn">Start Capture & Train</button>
            </div>`;
        setupGestureInput('gesture-tags-container');
    }

    async function loadExistingModelsUI() {
        const models = await (await fetch('/api/models')).json();
        const modelCards = models.map((m) => `
            <div class="model-card" data-model="${m}">
                <h3>${m}</h3>
                <div class="model-actions">
                    <button class="retrain-btn" data-model="${m}">Retrain</button>
                    <button class="test-btn" data-model="${m}">Test</button>
                    <button class="delete-btn" data-model="${m}">Delete</button>
                </div>
            </div>`).join('');
        dynamicContent.innerHTML = `
            <div class="content-section">
                <h2>Existing Models</h2>
                <div class="model-grid">${modelCards || '<p>No models found.</p>'}</div>
            </div>`;
    }

    function showTrainingConfirmation(modelName, isRetraining, capturedDataCount) {
        dynamicContent.innerHTML = `
            <div class="content-section">
                <h2>Confirm Training</h2>
                <p>Data capture complete for model: <b>${modelName}</b>.</p>
                <p>Captured <b>${capturedDataCount}</b> data points.</p>
                <button id="confirm-train-btn" data-model-name="${modelName}" data-is-retraining="${isRetraining}" data-captured-data-count="${capturedDataCount}">Yes, Train Model</button>
                <button id="retake-data-btn" data-model-name="${modelName}">No, Retake Data</button>
            </div>`;
    }

    // --- Action Functions ---
    function startNewCapture() {
        const modelName = document.getElementById('model-name-input').value.trim();
        const gestures = Array.from(document.querySelectorAll('#gesture-tags-container .tags-display .tag')).map((tag) => tag.textContent.replace('x', '').trim());
        const captureTime = document.getElementById('capture-time-select').value;

        if (!modelName || gestures.length === 0) {
            return alert('Please provide a model name and at least one gesture.');
        }

        fetch('/api/models')
            .then(res => res.json())
            .then(existingModels => {
                if (existingModels.includes(modelName)) {
                    return alert(`Model name '${modelName}' already exists.`);
                }
                startCapture(modelName, gestures, false, captureTime);
            });
    }

    function startCapture(modelName, gestures, isRetraining, captureTime) {
        statusMessage.innerText = 'Preparing for capture...';
        socket.emit('start_capture', { model_name: modelName, gestures, is_retraining: isRetraining, capture_time: captureTime });
    }

    async function startTest(modelName) {
        const { gestures } = await (await fetch(`/api/models/${modelName}`)).json();
        dynamicContent.innerHTML = `
            <div class="content-section">
                <h2>Testing: ${modelName}</h2>
                <p><b>Gestures:</b> ${gestures.join(', ')}</p>
                <button id="stop-test-btn">Stop Test</button>
            </div>`;
        socket.emit('start_test', { model_name: modelName });
    }

    // --- Helper: Gesture Input ---
    function setupGestureInput(containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = `
            <div class="tag-input-container">
                <div class="tags-display"></div>
                <button id="add-gesture-btn">+</button>
            </div>`;
        container.querySelector('#add-gesture-btn').addEventListener('click', () => {
            const gestureName = prompt("Enter new gesture name:");
            if (gestureName && gestureName.trim() !== '') {
                addGestureTag(gestureName.trim(), container);
            }
        });
    }

    function addGestureTag(gestureName, container) {
        const tagsDisplay = container.querySelector('.tags-display');
        const tag = document.createElement('div');
        tag.className = 'tag';
        tag.textContent = gestureName;
        const removeBtn = document.createElement('span');
        removeBtn.textContent = 'x';
        removeBtn.addEventListener('click', () => tag.remove());
        tag.appendChild(removeBtn);
        tagsDisplay.appendChild(tag);
    }

    // --- Initial Load ---
    updateModelCount();
    loadWelcomeUI();
    setupCameraControls();
});

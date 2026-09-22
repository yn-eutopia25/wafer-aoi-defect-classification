/* ===== annotator.js — AOI 标注工具核心逻辑 ===== */

// -------------------- 全局状态 --------------------
const state = {
    // 配置 (来自 /api/config)
    config: null,

    // 图片列表 (来自 /api/images)，每项含 image_id, filename, width, height, status, index
    imageList: [],
    currentIdx: 0,       // 当前图片在 imageList 中的索引

    // 当前图片对象
    img: new Image(),     // HTML Image 元素
    scale: 1.0,           // image_to_canvas 缩放比 = canvasW / imageW

    // 标注相关
    annotations: [],      // 当前图片的标注实例列表 [{...}, ...]
    selectedAnnId: null,  // 当前选中的 ann_id
    activeCategory: '',   // 当前选中类别 (A/B/C/D/E/F/N1/N2/N3)

    // 绘制状态
    drawMode: 'rectangle',   // 'rectangle' | 'polygon'
    drawing: false,           // 是否正在拖拽 (rectangle) / 是否在添加点 (polygon)
    drawStartX: 0,
    drawStartY: 0,
    drawEndX: 0,
    drawEndY: 0,
    hasPendingBox: false,     // rectangle 模式下有未保存的临时框
    polygonPoints: [],        // polygon 模式下已添加的 canvas 坐标点 [{cx,cy}, ...]
    polygonPreviewX: 0,       // polygon 模式下鼠标预览 canvas X
    polygonPreviewY: 0,       // polygon 模式下鼠标预览 canvas Y
};

// DOM 引用
const canvas = document.getElementById('image-canvas');
const ctx = canvas.getContext('2d');

// -------------------- 坐标转换 --------------------
/**
 * 图片像素坐标 → Canvas 显示坐标
 * @param {number} ix - 图片像素 X
 * @param {number} iy - 图片像素 Y
 * @returns {{cx: number, cy: number}}
 */
function imageToCanvas(ix, iy) {
    return { cx: ix * state.scale, cy: iy * state.scale };
}

/**
 * Canvas 显示坐标 → 图片像素坐标 (clamp 到图片范围内)
 * @param {number} cx - Canvas X
 * @param {number} cy - Canvas Y
 * @returns {{ix: number, iy: number}}
 */
function canvasToImage(cx, cy) {
    const ix = Math.round(cx / state.scale);
    const iy = Math.round(cy / state.scale);
    return {
        ix: Math.max(0, Math.min(ix, state.img.naturalWidth - 1)),
        iy: Math.max(0, Math.min(iy, state.img.naturalHeight - 1))
    };
}

/**
 * 获取当前临时标注的原图像素坐标
 * - rectangle: bbox [x_min, y_min, x_max, y_max]
 * - polygon: 图片像素点数组 [[ix,iy], ...] 或 null
 */
function getPendingImageCoords() {
    if (state.drawMode === 'polygon' && state.polygonPoints.length >= 3) {
        return state.polygonPoints.map(p => {
            const img = canvasToImage(p.cx, p.cy);
            return [img.ix, img.iy];
        });
    }
    if (state.drawMode === 'rectangle' && state.hasPendingBox) {
        const sx = Math.min(state.drawStartX, state.drawEndX);
        const sy = Math.min(state.drawStartY, state.drawEndY);
        const ex = Math.max(state.drawStartX, state.drawEndX);
        const ey = Math.max(state.drawEndY, state.drawStartY);
        const p1 = canvasToImage(sx, sy);
        const p2 = canvasToImage(ex, ey);
        return [p1.ix, p1.iy, p2.ix, p2.iy];  // bbox
    }
    return null;
}

// -------------------- API 调用 --------------------
const API = {
    async get(url) {
        const r = await fetch(url);
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },
    async post(url, data) {
        const r = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },
    async del(url) {
        const r = await fetch(url, { method: 'DELETE' });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    }
};

// -------------------- 初始化 --------------------
async function init() {
    try {
        state.config = await API.get('/api/config');
        state.imageList = await API.get('/api/images');
        const start = await API.get('/api/start');

        // 确定起始 index
        let startIdx = 0;
        if (start.image_id) {
            startIdx = state.imageList.findIndex(item => item.image_id === start.image_id);
            if (startIdx < 0) startIdx = 0;
        }
        state.currentIdx = startIdx;

        renderCategoryButtons();
        renderLegend();
        renderAttributeForm();   // 填充 region 下拉等
        await loadCurrentImage();
        setupEventListeners();
        setupKeyboard();
    } catch (e) {
        showError('初始化失败: ' + e.message);
    }
}

// -------------------- 属性表单初始化 --------------------
function renderAttributeForm() {
    // Region 下拉（默认 surface）
    const regionSelect = document.getElementById('prop-region');
    regionSelect.innerHTML = '<option value="">--</option>';
    const regions = state.config.regions || [];
    for (const r of regions) {
        const opt = document.createElement('option');
        opt.value = r;
        opt.textContent = r;
        if (r === 'surface') opt.selected = true;
        regionSelect.appendChild(opt);
    }
}

// -------------------- 类别按钮 --------------------
function renderCategoryButtons() {
    const defectGrid = document.getElementById('defect-buttons');
    const normalGrid = document.getElementById('normal-buttons');

    [defectGrid, normalGrid].forEach(el => el.innerHTML = '');

    const defectClasses = state.config.defect_classes || {};
    for (const [code, info] of Object.entries(defectClasses)) {
        const btn = createCatButton(code, info.display_name, `cat-${code}`);
        defectGrid.appendChild(btn);
    }

    const normalClasses = state.config.normal_classes || {};
    for (const [code, info] of Object.entries(normalClasses)) {
        const btn = createCatButton(code, info.display_name, `cat-${code}`);
        normalGrid.appendChild(btn);
    }
}

function createCatButton(code, displayName, cssClass) {
    const btn = document.createElement('button');
    btn.className = `cat-btn ${cssClass}`;
    btn.textContent = code + '\n' + displayName;
    btn.style.whiteSpace = 'pre-line';
    btn.addEventListener('click', () => selectCategory(code));
    return btn;
}

function selectCategory(code) {
    state.activeCategory = code;
    // 取消所有激活
    document.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
    // 选中当前
    const matchClass = `cat-${code}`;
    document.querySelectorAll(`.${matchClass}`).forEach(b => b.classList.add('active'));
    document.getElementById('selected-cat').textContent = code;
}

function renderLegend() {
    const legend = document.getElementById('legend');
    legend.innerHTML = '';

    const defectClasses = state.config.defect_classes || {};
    const normalClasses = state.config.normal_classes || {};

    for (const [code, info] of Object.entries(defectClasses)) {
        legend.appendChild(createLegendItem(code, info.display_name, `cat-${code}`));
    }
    for (const [code, info] of Object.entries(normalClasses)) {
        legend.appendChild(createLegendItem(code, info.display_name, `cat-${code}`));
    }
}

function createLegendItem(code, name, cssClass) {
    const div = document.createElement('div');
    div.className = 'legend-row';
    // 取 CSS 变量颜色
    const el = document.createElement('span');
    el.className = `cat-btn ${cssClass}`;
    el.style.cssText = 'display:inline-block; width:16px; height:12px; padding:0; border-radius:2px; cursor:default; transform:none;';
    const dot = document.createElement('span');
    dot.className = 'legend-dot';
    // 用 cat-btn 的颜色从 DOM 读取
    const colorMap = {
        'A':'#ff6b6b','B':'#ffa94d','C':'#ffd43b','D':'#69db7c',
        'E':'#74c0fc','F':'#da77f2','G':'#adb5bd',
        'N1':'#38d9a9','N2':'#4dabf7','N3':'#b197fc'
    };
    dot.style.background = colorMap[code] || '#888';
    div.appendChild(dot);
    div.appendChild(document.createTextNode(`${code}: ${name}`));
    return div;
}

// -------------------- 图片加载 --------------------
async function loadCurrentImage() {
    const item = state.imageList[state.currentIdx];
    if (!item) return;

    // 更新顶部信息
    document.getElementById('info-image-id').textContent = item.image_id;
    document.getElementById('info-filename').textContent = item.filename;
    document.getElementById('info-progress').textContent = `${state.currentIdx + 1} / ${state.imageList.length}`;
    document.getElementById('info-status').textContent = item.status;

    // 更新进度条
    const doneCount = state.imageList.filter(i => i.status === 'done').length;
    const pct = (doneCount / state.imageList.length * 100).toFixed(1);
    document.getElementById('progress-fill').style.width = pct + '%';
    document.getElementById('info-pct').textContent = pct + '%';

    // 加载图片
    state.img.onload = () => {
        fitCanvas();
        loadAnnotations();
    };
    state.img.src = `/api/image/${item.image_id}`;
}

/** 根据窗口大小适配 Canvas 尺寸 */
function fitCanvas() {
    const panel = document.getElementById('canvas-panel');
    const maxW = panel.clientWidth - 20;
    const maxH = panel.clientHeight - 20;
    const iw = state.img.naturalWidth;
    const ih = state.img.naturalHeight;
    state.scale = Math.min(maxW / iw, maxH / ih, 1.0);
    canvas.width = Math.round(iw * state.scale);
    canvas.height = Math.round(ih * state.scale);
    document.getElementById('zoom-tag').textContent = Math.round(state.scale * 100) + '%';
}

/** 加载当前图片的标注 */
async function loadAnnotations() {
    const item = state.imageList[state.currentIdx];
    try {
        const data = await API.get(`/api/annotations/${item.image_id}`);
        state.annotations = data.annotations || [];
    } catch (e) {
        state.annotations = [];
    }
    state.selectedAnnId = null;
    state.hasPendingBox = false;
    state.drawing = false;
    state.polygonPoints = [];
    renderAll();
}

// -------------------- 渲染 --------------------
function renderAll() {
    renderCanvas();
    renderAnnotationTable();
}

function renderCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);

    // 绘制已保存的标注
    for (const ann of state.annotations) {
        if (ann.geometry_type === 'polygon') {
            drawPolygonAnnotation(ann, ann.ann_id === state.selectedAnnId);
        } else {
            drawRectAnnotation(ann, ann.ann_id === state.selectedAnnId);
        }
    }

    // ----- 绘制临时框 (rectangle 模式) -----
    if (state.drawMode === 'rectangle' && state.hasPendingBox) {
        const x = Math.min(state.drawStartX, state.drawEndX);
        const y = Math.min(state.drawStartY, state.drawEndY);
        const w = Math.abs(state.drawEndX - state.drawStartX);
        const h = Math.abs(state.drawEndY - state.drawStartY);
        const color = getCatColor(state.activeCategory);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }

    // ----- 绘制临时 polygon (polygon 模式) -----
    if (state.drawMode === 'polygon' && state.polygonPoints.length > 0) {
        const color = getCatColor(state.activeCategory);
        const pts = state.polygonPoints;

        // 顶点之间的边（实线）
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(pts[0].cx, pts[0].cy);
        for (let i = 1; i < pts.length; i++) {
            ctx.lineTo(pts[i].cx, pts[i].cy);
        }
        // 如果已完成 (drawing=false)，闭合到第一个点
        if (!state.drawing && pts.length >= 3) {
            ctx.closePath();

            // 半透明填充
            ctx.fillStyle = hexToRgba(color, 0.15);
            ctx.fill();
        }
        ctx.stroke();

        // 预览线（仅在绘制中显示，完成后隐藏）
        if (state.drawing && pts.length >= 1) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(pts[pts.length - 1].cx, pts[pts.length - 1].cy);
            ctx.lineTo(state.polygonPreviewX, state.polygonPreviewY);
            ctx.stroke();
            ctx.setLineDash([]);

            // 闭合虚线（≥2 点时显示到首点的虚线）
            if (pts.length >= 2) {
                ctx.strokeStyle = hexToRgba(color, 0.5);
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 4]);
                ctx.beginPath();
                ctx.moveTo(pts[pts.length - 1].cx, pts[pts.length - 1].cy);
                ctx.lineTo(pts[0].cx, pts[0].cy);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }

        // 顶点圆点
        for (const p of pts) {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(p.cx, p.cy, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

function drawRectAnnotation(ann, isSelected) {
    const bbox = ann.bbox;
    if (!bbox || bbox.length < 4) return;

    const [ix_min, iy_min, ix_max, iy_max] = bbox;
    const p1 = imageToCanvas(ix_min, iy_min);
    const p2 = imageToCanvas(ix_max, iy_max);
    const cx = p1.cx, cy = p1.cy, cw = p2.cx - p1.cx, ch = p2.cy - p1.cy;

    const color = getCatColor(ann.label_code);

    // 半透明填充
    ctx.fillStyle = isSelected ? 'rgba(255,255,255,0.2)' : hexToRgba(color, 0.15);
    ctx.fillRect(cx, cy, cw, ch);

    // 边框
    ctx.strokeStyle = isSelected ? '#fff' : color;
    ctx.lineWidth = isSelected ? 2.5 : 2;
    ctx.strokeRect(cx, cy, cw, ch);

    // 标签
    const shortId = ann.ann_id ? ann.ann_id.slice(-4) : '';
    const label = `${ann.label_code} ${shortId}`;
    const fontSize = Math.max(10, 12 * state.scale);
    ctx.font = `${fontSize}px system-ui`;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(cx, cy - 16 * state.scale, textWidth + 4, 16 * state.scale);
    ctx.fillStyle = isDarkColor(color) ? '#fff' : '#111';
    ctx.fillText(label, cx + 2, cy - 4 * state.scale);

    // 选中高亮
    if (isSelected) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 3;
        ctx.setLineDash([4, 2]);
        ctx.strokeRect(cx - 2, cy - 2, cw + 4, ch + 4);
        ctx.setLineDash([]);
    }
}

/** 绘制已保存的 polygon 标注 */
function drawPolygonAnnotation(ann, isSelected) {
    const points = ann.points;
    if (!points || points.length < 3) return;

    const color = getCatColor(ann.label_code);
    const canvasPoints = points.map(p => {
        const c = imageToCanvas(p[0], p[1]);
        return { cx: c.cx, cy: c.cy };
    });

    // 半透明填充
    ctx.fillStyle = isSelected ? 'rgba(255,255,255,0.2)' : hexToRgba(color, 0.15);
    ctx.beginPath();
    ctx.moveTo(canvasPoints[0].cx, canvasPoints[0].cy);
    for (let i = 1; i < canvasPoints.length; i++) {
        ctx.lineTo(canvasPoints[i].cx, canvasPoints[i].cy);
    }
    ctx.closePath();
    ctx.fill();

    // 轮廓
    ctx.strokeStyle = isSelected ? '#fff' : color;
    ctx.lineWidth = isSelected ? 2.5 : 2;
    ctx.stroke();

    // 顶点
    for (const p of canvasPoints) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.cx, p.cy, 3, 0, Math.PI * 2);
        ctx.fill();
    }

    // 标签
    const bbox = ann.bbox;
    if (bbox && bbox.length === 4) {
        const c0 = imageToCanvas(bbox[0], bbox[1]);
        const shortId = ann.ann_id ? ann.ann_id.slice(-4) : '';
        const label = `${ann.label_code} ${shortId}`;
        const fontSize = Math.max(10, 12 * state.scale);
        ctx.font = `${fontSize}px system-ui`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(c0.cx, c0.cy - 16 * state.scale, textWidth + 4, 16 * state.scale);
        ctx.fillStyle = isDarkColor(color) ? '#fff' : '#111';
        ctx.fillText(label, c0.cx + 2, c0.cy - 4 * state.scale);
    }
}

function renderAnnotationTable() {
    const tbody = document.getElementById('ann-tbody');
    const countEl = document.getElementById('ann-count');

    tbody.innerHTML = '';
    countEl.textContent = `(${state.annotations.length})`;

    for (const ann of state.annotations) {
        const tr = document.createElement('tr');
        if (ann.ann_id === state.selectedAnnId) tr.classList.add('selected');
        tr.addEventListener('click', () => selectAnnotation(ann.ann_id));

        const bbox = ann.bbox || [0, 0, 0, 0];
        const bboxStr = `[${bbox[0]},${bbox[1]},${bbox[2]},${bbox[3]}]`;

        const color = getCatColor(ann.label_code);
        tr.innerHTML = `
            <td><span class="ann-color-dot" style="background:${color}"></span>${ann.ann_id || ''}</td>
            <td>${ann.label_code || ''}</td>
            <td>${ann.label || ''}</td>
            <td>${ann.region || '-'}</td>
            <td>${ann.severity || '-'}</td>
            <td>${bboxStr}</td>
            <td>${ann.note || '-'}</td>
        `;
        tbody.appendChild(tr);
    }
}

function selectAnnotation(annId) {
    state.selectedAnnId = annId;
    state.hasPendingBox = false;
    state.drawing = false;
    state.polygonPoints = [];
    renderAll();
}

// -------------------- Canvas 交互 --------------------
function setupEventListeners() {
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('mouseleave', onMouseUp);
    canvas.addEventListener('dblclick', onDoubleClick);  // polygon: 双击完成
    window.addEventListener('resize', () => { fitCanvas(); renderCanvas(); });
}

function onMouseDown(e) {
    if (!state.activeCategory) {
        alert('请先选择一个类别！');
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const cx = Math.max(0, Math.min(e.clientX - rect.left, canvas.width));
    const cy = Math.max(0, Math.min(e.clientY - rect.top, canvas.height));

    if (state.drawMode === 'rectangle') {
        // --- rectangle 模式：开始拖拽 ---
        state.drawing = true;
        state.hasPendingBox = true;
        state.drawStartX = cx;
        state.drawStartY = cy;
        state.drawEndX = cx;
        state.drawEndY = cy;
        state.selectedAnnId = null;
        renderAll();

    } else if (state.drawMode === 'polygon') {
        // --- polygon 模式：添加点 ---
        state.drawing = true;
        state.polygonPoints.push({ cx, cy });
        state.polygonPreviewX = cx;
        state.polygonPreviewY = cy;
        state.selectedAnnId = null;
        renderAll();
    }
}

function onDoubleClick(e) {
    if (state.drawMode !== 'polygon') return;
    // 双击 = Enter = 完成 polygon
    finishPolygon();
}

function onMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const cx = Math.max(0, Math.min(e.clientX - rect.left, canvas.width));
    const cy = Math.max(0, Math.min(e.clientY - rect.top, canvas.height));

    if (state.drawMode === 'rectangle') {
        if (!state.drawing) return;
        state.drawEndX = cx;
        state.drawEndY = cy;
        renderAll();

    } else if (state.drawMode === 'polygon') {
        state.polygonPreviewX = cx;
        state.polygonPreviewY = cy;
        if (state.drawing && state.polygonPoints.length > 0) {
            renderAll();
        }
    }
}

function onMouseUp(e) {
    if (state.drawMode === 'rectangle' && state.drawing) {
        state.drawing = false;
        // 保留临时框等待用户点击"保存"
    }
    // polygon 模式下 mouseup 不做任何事，点已通过 mousedown 添加
}

/** 完成 polygon 绘制 */
function finishPolygon() {
    if (state.polygonPoints.length < 3) {
        alert('polygon 至少需要 3 个点才能完成。');
        return;
    }
    // 标记为有 pending polygon 等待保存
    state.drawing = false;
    renderAll();  // 刷新画布，移除预览线，显示"已完成"状态
}

// -------------------- 按钮操作 --------------------
async function saveCurrentAnnotation() {
    const item = state.imageList[state.currentIdx];
    if (!item) return;

    if (!state.activeCategory) {
        alert('请先选择一个类别。');
        return;
    }

    let body;

    if (state.drawMode === 'rectangle') {
        // -- rectangle 模式 --
        if (!state.hasPendingBox) {
            alert('没有待保存的标注框。');
            return;
        }
        const coords = getPendingImageCoords();
        if (!coords || coords.length !== 4) return;
        const [x_min, y_min, x_max, y_max] = coords;
        const w = x_max - x_min;
        const h = y_max - y_min;
        if (w < 5 || h < 5) {
            alert('框太小 (宽或高小于 5 px)，不能保存。');
            return;
        }
        body = {
            label_code: state.activeCategory,
            geometry_type: 'rectangle',
            bbox: [x_min, y_min, x_max, y_max],
        };

    } else if (state.drawMode === 'polygon') {
        // -- polygon 模式 --
        if (state.polygonPoints.length < 3) {
            alert(`polygon 至少需要 3 个点，当前只有 ${state.polygonPoints.length} 个。`);
            return;
        }
        const coords = getPendingImageCoords();  // [[ix,iy], ...]
        if (!coords || coords.length < 3) return;
        body = {
            label_code: state.activeCategory,
            geometry_type: 'polygon',
            points: coords,
        };

    } else {
        alert('未知绘制模式: ' + state.drawMode);
        return;
    }

    // 通用属性
    body.region = document.getElementById('prop-region').value;
    body.severity = parseInt(document.getElementById('prop-severity').value) || null;
    body.quality = document.getElementById('prop-quality').value || 'good';
    body.note = document.getElementById('prop-note').value || '';

    try {
        const result = await API.post(`/api/annotations/${item.image_id}`, body);
        // 清除临时绘制状态
        state.hasPendingBox = false;
        state.polygonPoints = [];
        state.drawing = false;
        await loadAnnotations();
        item.status = 'in_progress';
        document.getElementById('info-status').textContent = 'in_progress';
    } catch (e) {
        showError('保存失败: ' + e.message);
    }
}

function undoPending() {
    state.hasPendingBox = false;
    state.drawing = false;
    state.polygonPoints = [];
    renderAll();
}

async function deleteSelectedAnnotation() {
    if (!state.selectedAnnId) {
        alert('请先在标注列表中选中一个标注。');
        return;
    }
    if (!confirm('确认删除此标注？')) return;

    const item = state.imageList[state.currentIdx];
    try {
        await API.del(`/api/annotations/${item.image_id}/${state.selectedAnnId}`);
        state.selectedAnnId = null;
        await loadAnnotations();
    } catch (e) {
        showError('删除失败: ' + e.message);
    }
}

async function setImageStatus(status) {
    const item = state.imageList[state.currentIdx];
    try {
        await API.post(`/api/image_status/${item.image_id}`, { status });
        item.status = status;
        document.getElementById('info-status').textContent = status;
    } catch (e) {
        showError('状态更新失败: ' + e.message);
    }
}

async function navPrev() {
    if (state.currentIdx > 0) {
        state.currentIdx--;
        await loadCurrentImage();
    }
}

async function navNext() {
    if (state.currentIdx < state.imageList.length - 1) {
        state.currentIdx++;
        await loadCurrentImage();
    }
}

async function saveAndRefresh() {
    await loadCurrentImage();
}

// -------------------- 键盘快捷键 --------------------
function setupKeyboard() {
    document.addEventListener('keydown', (e) => {
        // 忽略输入框中的按键
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;

        const key = e.key.toLowerCase();

        // 模式切换
        if (key === 'r') { e.preventDefault(); switchDrawMode('rectangle'); }
        else if (key === 'p') { e.preventDefault(); switchDrawMode('polygon'); }

        // 类别选择
        else if (key === 'a') selectCategory('A');
        else if (key === 'b' && !e.ctrlKey && !e.metaKey) selectCategory('B');
        else if (key === 'c') selectCategory('C');
        else if (key === 'd' && !e.ctrlKey && !e.metaKey) selectCategory('D');
        else if (key === 'e') selectCategory('E');
        else if (key === 'f') selectCategory('F');
        else if (key === 'g') selectCategory('G');

        // Severity
        else if (key === '1') document.getElementById('prop-severity').value = '1';
        else if (key === '2') document.getElementById('prop-severity').value = '2';
        else if (key === '3') document.getElementById('prop-severity').value = '3';

        // 操作
        else if (key === 's') { e.preventDefault(); saveCurrentAnnotation(); }
        else if (key === 'enter' && state.drawMode === 'polygon') { e.preventDefault(); finishPolygon(); }
        else if (key === 'escape') undoPending();
        else if (key === 'backspace' && state.drawMode === 'polygon') {
            e.preventDefault();
            if (state.polygonPoints.length > 0) {
                state.polygonPoints.pop();
                renderAll();
            }
        }
        else if (key === 'delete') { e.preventDefault(); deleteSelectedAnnotation(); }
        else if (key === 'n') { e.preventDefault(); navNext(); }

        // 用 ArrowLeft/ArrowRight 导航
        else if (key === 'arrowleft') { e.preventDefault(); navPrev(); }
        else if (key === 'arrowright') { e.preventDefault(); navNext(); }
    });

    // 单独处理大写 B → 上一张
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'B' && !e.ctrlKey && !e.altKey && !e.metaKey && e.shiftKey) {
            e.preventDefault();
            navPrev();
        }
    });
}

/** 切换绘制模式 */
function switchDrawMode(mode) {
    state.drawMode = mode;
    state.hasPendingBox = false;
    state.drawing = false;
    state.polygonPoints = [];

    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active-mode'));
    const activeBtn = document.getElementById('mode-' + mode);
    if (activeBtn) activeBtn.classList.add('active-mode');

    document.getElementById('draw-mode-tag').textContent = mode === 'polygon' ? 'POLYGON' : 'RECT';
    renderAll();
}

// -------------------- 工具函数 --------------------
function getCatColor(code) {
    const map = {
        'A':'#ff6b6b','B':'#ffa94d','C':'#ffd43b','D':'#69db7c',
        'E':'#74c0fc','F':'#da77f2','G':'#adb5bd',
        'N1':'#38d9a9','N2':'#4dabf7','N3':'#b197fc'
    };
    return map[code] || '#fff';
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

function isDarkColor(hex) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return (r * 0.299 + g * 0.587 + b * 0.114) < 150;
}

function showError(msg) {
    alert(msg);
    console.error(msg);
}

// -------------------- 启动 --------------------
window.addEventListener('DOMContentLoaded', init);

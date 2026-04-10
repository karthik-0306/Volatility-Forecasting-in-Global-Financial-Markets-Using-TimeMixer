let masterTickers = {};
let currentChart = null;
let currentForecastData = null; // Store data globally to parse for single-day inspections

// Initialization
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch('/api/tickers');
        const data = await res.json();
        masterTickers = data.assets;
        loadTickers();
    } catch (e) {
        console.error("Failed to load tickers:", e);
    }
    
    // Horizon Button Toggles
    const buttons = document.querySelectorAll('.hz-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            buttons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            if (currentChart) {
                triggerForecast();
            }
        });
    });
});

function loadTickers() {
    const acSelect = document.getElementById('asset-class').value;
    const tSelect = document.getElementById('ticker-select');
    tSelect.innerHTML = '';
    
    let toLoad = [];
    if (acSelect === 'all') {
        Object.values(masterTickers).forEach(arr => toLoad = toLoad.concat(arr));
    } else {
        toLoad = masterTickers[acSelect] || [];
    }
    
    toLoad.sort().forEach(ticker => {
        const opt = document.createElement('option');
        opt.value = ticker;
        opt.textContent = ticker;
        if (ticker === 'AAPL') opt.selected = true;
        tSelect.appendChild(opt);
    });
}

async function triggerForecast() {
    const ticker = document.getElementById('ticker-select').value;
    const acNode = document.getElementById('asset-class').value;
    
    let targetAssetClass = acNode;
    if (targetAssetClass === 'all') {
        for (const [ac, arr] of Object.entries(masterTickers)) {
            if (arr.includes(ticker)) {
                targetAssetClass = ac;
                break;
            }
        }
    }
    
    const horizon = document.querySelector('.hz-btn.active').dataset.val;
    
    document.getElementById('display-ticker').textContent = `${ticker} (Loading...)`;
    document.getElementById('display-ticker').style.color = '#94a3b8';
    
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: ticker,
                horizon: parseInt(horizon),
                asset_class: targetAssetClass
            })
        });
        
        if (!res.ok) throw new Error((await res.json()).detail);
        
        currentForecastData = await res.json();
        const data = currentForecastData;
        
        document.getElementById('display-ticker').textContent = ticker;
        document.getElementById('display-ticker').style.color = 'var(--accent-glow)';
        
        // Populate the Daily Inspector Dropdown
        const dateSelect = document.getElementById('date-select');
        dateSelect.innerHTML = '';
        data.forecast.dates.forEach((d, i) => {
            const opt = document.createElement('option');
            opt.value = i; // Store index
            opt.textContent = d;
            dateSelect.appendChild(opt);
        });
        
        renderChart(data);
        
        // By default, inspect the very first future day
        inspectSpecificDate();
        
    } catch(e) {
        alert("Error fetching forecast: " + e.message);
        document.getElementById('display-ticker').textContent = 'Error';
    }
}

// ─── DAILY INSPECTOR LOGIC (V2 Update) ─────────────────────────── //

function inspectSpecificDate(idx = null) {
    if (!currentForecastData) return;
    
    const dateSelect = document.getElementById('date-select');
    if (idx !== null) {
        dateSelect.value = idx; // sync dropdown if clicked from graph
    }
    const selectedIdx = dateSelect.value;
    
    const val = currentForecastData.forecast.values[selectedIdx];
    const baseline = currentForecastData.baseline;
    
    document.getElementById('val-baseline').textContent = baseline.toFixed(4);
    document.getElementById('val-projected').textContent = val.toFixed(4);
    
    const delta = ((val - baseline) / baseline) * 100;
    const deltaEl = document.getElementById('val-delta');
    deltaEl.textContent = `${delta > 0 ? '+' : ''}${delta.toFixed(1)}%`;
    
    const badge = document.getElementById('risk-badge');
    if (delta > 5) {
        deltaEl.style.color = 'var(--danger)';
        badge.className = 'badge danger';
        badge.innerHTML = '🔴 HIGH RISK';
    } else if (delta < -5) {
        deltaEl.style.color = 'var(--safe)';
        badge.className = 'badge safe';
        badge.innerHTML = '🟢 LOW RISK';
    } else {
        deltaEl.style.color = 'var(--neutral)';
        badge.className = 'badge neutral';
        badge.innerHTML = '🟡 NORMAL';
    }
}

// ─── CHARTJS VISUALIZATION LOGIC ───────────────────────────────── //

function renderChart(data) {
    const ctx = document.getElementById('volatilityChart').getContext('2d');
    if (currentChart) currentChart.destroy();
    
    const allLabels = data.history.dates.concat(data.forecast.dates);
    const histLen = data.history.values.length;
    
    const histData = data.history.values.concat(Array(data.forecast.values.length).fill(null));
    const foreData = Array(histLen - 1).fill(null);
    foreData.push(data.history.values[histLen - 1]);
    data.forecast.values.forEach(v => foreData.push(v));
    
    const baseline = data.baseline;
    
    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allLabels,
            datasets: [
                {
                    label: 'Historical Actuals (Yang-Zhang)',
                    data: histData,
                    borderColor: 'rgba(255, 255, 255, 0.25)',
                    borderWidth: 2,
                    pointRadius: 0, // Removes messy dots
                    pointHoverRadius: 5,
                    tension: 0.1
                },
                {
                    label: 'TimeMixer Projection',
                    data: foreData,
                    borderColor: '#38bdf8', 
                    backgroundColor: 'rgba(56, 189, 248, 0.15)',
                    borderWidth: 3,
                    pointRadius: 0, // Removes messy dots
                    pointHoverRadius: 8, // Large glow on hover
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#38bdf8',
                    pointHoverBorderWidth: 3,
                    fill: true,
                    tension: 0.4 // Extra smooth mathematics curve
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const dataIndex = elements[0].index;
                    // Check if clicked point is in the future
                    if (dataIndex >= histLen) {
                        const targetIdx = dataIndex - histLen;
                        inspectSpecificDate(targetIdx);
                    }
                }
            },
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } },
                tooltip: { 
                    backgroundColor: 'rgba(11, 15, 25, 0.95)', 
                    titleFont: { family: 'Outfit', size: 14 }, 
                    bodyFont: { family: 'Outfit', size: 14 },
                    padding: 12,
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(4);
                            }
                            
                            // Inject Custom Text into Tooltip for Future Forecasts
                            if (context.datasetIndex === 1 && context.dataIndex >= histLen) {
                                const val = context.parsed.y;
                                if (val > baseline * 1.05) {
                                    label += '  [🔴 HIGH RISK]';
                                } else if (val < baseline * 0.95) {
                                    label += '  [🟢 LOW RISK]';
                                } else {
                                    label += '  [🟡 NORMAL]';
                                }
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#64748b', maxTicksLimit: 12 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: { color: '#64748b' },
                    title: { display: true, text: 'Annualized Volatility', color: '#94a3b8'}
                }
            }
        }
    });
}

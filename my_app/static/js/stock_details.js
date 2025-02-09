let stockSymbol = document.querySelector("h2").innerText;
let chartInstance = null;

// Fetch stock data and update the chart
function fetchStockData(range) {
    fetch(`/api/stock_data/${stockSymbol}?range=${range}`)
        .then(response => response.json())
        .then(data => {
            if (data.error || !data.timestamps || !data.prices || data.prices.length === 0) {
                console.error("Error fetching stock data or empty dataset:", data.error || "No data available");
                return;
            }

            const labels = data.timestamps;  // Full timestamps for smooth graph
            const prices = data.prices;
            const xLabels = data.x_labels;  // Optimized X-axis labels

            const ctx = document.getElementById('stockChart').getContext('2d');

            // Destroy previous chart if it exists
            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: `${stockSymbol} Price`,
                        data: prices,
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            display: true,
                            title: { display: true, text: "Time" },
                            ticks: {
                                callback: function(value, index) {
                                    return xLabels.includes(labels[index]) ? labels[index] : "";
                                },
                                autoSkip: false,
                                maxRotation: 45,
                                minRotation: 0
                            }
                        },
                        y: {
                            display: true,
                            title: { display: true, text: "Price (INR)" }
                        }
                    },
                    elements: {
                        line: { tension: 0.2 }
                    },
                    animation: {
                        duration: 800
                    }
                }
            });
        })
        .catch(error => console.error("Error fetching stock data:", error));
}

// Update chart when a new range is selected
function updateChart(range) {
    document.querySelectorAll('.filters button').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.filters button[onclick="updateChart('${range}')"]`).classList.add('active');
    fetchStockData(range);
}

// Fetch real-time stock price
function fetchLivePrice() {
    fetch(`/api/live_price/${stockSymbol}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error("Error fetching live price:", data.error);
                return;
            }

            document.getElementById('currentPrice').innerText = data.price || "N/A";
            document.getElementById('marketStatus').innerText = `(${data.market_status})`;
        })
        .catch(error => console.error("Error fetching live price:", error));
}

// Fetch the range price for the selected model
function updateForecast() {
    let selectedModel = document.getElementById("model").value;

    // Send AJAX request to get the range price for the selected model
    fetch(`/stock/{{ stock }}/?model=${selectedModel}`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.range_price) {
            document.getElementById("rangePrice").innerText = `${data.range_price} INR`;
        } else {
            document.getElementById("rangePrice").innerText = "N/A";
        }
    })
    .catch(error => {
        console.error("Error fetching forecast:", error);
        document.getElementById("rangePrice").innerText = "Error";
    });
}

// Load initial data
fetchStockData('1D');
fetchLivePrice();
updateForecast();

// Auto-refresh live price every 30 seconds
setInterval(fetchLivePrice, 30000);

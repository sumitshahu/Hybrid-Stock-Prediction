document.addEventListener('DOMContentLoaded', () => {
    const stockSymbol = window.location.pathname.split('/')[2]; // Get stock symbol from URL
    const ctx = document.getElementById('priceChart').getContext('2d');
    let chart;

    function fetchHistoricalData(period = '1mo') {
        fetch(`/stock/${stockSymbol}/historical_data/?period=${period}`)
            .then(response => response.json())
            .then(data => {
                const labels = Object.keys(data.Close);
                const prices = Object.values(data.Close);
                
                if (chart) {
                    chart.destroy(); // Destroy existing chart before creating new one
                }

                chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Price',
                            data: prices,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    unit: 'day'
                                }
                            }
                        }
                    }
                });
            })
            .catch(error => console.error('Error fetching historical data:', error));
    }

    // Fetch initial data
    fetchHistoricalData();

    // Set event listeners for buttons (1d, 1w, 1m, etc.)
    document.querySelectorAll('.timeframe-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            fetchHistoricalData(btn.getAttribute('data-period'));
        });
    });
});

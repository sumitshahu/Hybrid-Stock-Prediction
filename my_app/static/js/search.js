document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('stockSearch');
    const stockList = document.getElementById('stockList');

    if (input) {
        input.addEventListener('input', () => {
            const query = input.value;

            if (query.length > 0) {
                fetch(`/search_stocks/?query=${query}`)
                    .then(response => response.json())
                    .then(data => {
                        stockList.innerHTML = '';
                        data.forEach(stock => {
                            const li = document.createElement('li');
                            li.textContent = stock;
                            li.addEventListener('click', () => {
                                window.location.href = `/stock/${encodeURIComponent(stock)}/`;
                            });
                            stockList.appendChild(li);
                        });

                        stockList.style.display = data.length > 0 ? 'block' : 'none';
                    })
                    .catch(error => console.error('Error fetching stocks:', error));
            } else {
                stockList.style.display = 'none';
            }
        });

        document.addEventListener('click', (event) => {
            if (!input.contains(event.target) && !stockList.contains(event.target)) {
                stockList.style.display = 'none';
            }
        });

        input.addEventListener('focus', () => {
            if (stockList.childElementCount > 0) {
                stockList.style.display = 'block';
            }
        });
    }

    // --- NEW CODE: Support for Tools Page ---
    const compareBtn = document.getElementById('compareButton');
    const compareInput = document.getElementById('compareStock');
    const chartContainer = document.getElementById('chartContainer');

    if (compareBtn && compareInput && chartContainer) {
        compareBtn.addEventListener('click', () => {
            const stockName = compareInput.value.trim();
            if (!stockName) {
                alert('Please enter a stock name');
                return;
            }

            fetch(`/compare_models/?stock=${encodeURIComponent(stockName)}`)
                .then(response => response.text())
                .then(html => {
                    chartContainer.innerHTML = html;
                })
                .catch(err => {
                    chartContainer.innerHTML = '<p>Error loading comparison charts.</p>';
                    console.error('Error:', err);
                });
        });
    }
});

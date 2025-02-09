document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('stockSearch');
    const stockList = document.getElementById('stockList');

    input.addEventListener('input', () => {
        const query = input.value;

        if (query.length > 0) {
            fetch(`/search_stocks/?query=${query}`)
                .then(response => response.json())
                .then(data => {
                    // Clear previous results
                    stockList.innerHTML = '';

                    // Populate the list with new results
                    data.forEach(stock => {
                        const li = document.createElement('li');
                        li.textContent = stock;

                        // Use Django URL pattern dynamically
                        li.addEventListener('click', () => {
                            window.location.href = `/stock/${encodeURIComponent(stock)}/`;
                        });

                        stockList.appendChild(li);
                    });

                    // Show the list if there are results
                    stockList.style.display = data.length > 0 ? 'block' : 'none';
                })
                .catch(error => console.error('Error fetching stocks:', error));
        } else {
            // Hide the list if query is empty
            stockList.style.display = 'none';
        }
    });

    // Hide the stock list when clicking outside of it
    document.addEventListener('click', (event) => {
        if (!input.contains(event.target) && !stockList.contains(event.target)) {
            stockList.style.display = 'none';
        }
    });

    // Prevent the stock list from hiding when clicking inside the input
    input.addEventListener('focus', () => {
        if (stockList.childElementCount > 0) {
            stockList.style.display = 'block';
        }
    });
});

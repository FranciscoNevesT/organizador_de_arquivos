function update(){
    fetch('/update')
    .then(response => {
        if (response.ok) {
            window.location.reload(); // Reload the page if fetch is successful
        } else {
            // Handle error if needed
            console.error('Error fetching data');
        }
    })
    .catch(error => {
        // Handle fetch error
        console.error('Fetch error:', error);
    });
}

function process_file(file_id){
    fetch('/catalogo/process/file/' + file_id)
    .then(response => {
        if (response.ok) {
            alert("Text processed"); // Reload the page if fetch is successful
        } else {
            // Handle error if needed
            console.error('Error fetching data');
        }
    })
    .catch(error => {
        // Handle fetch error
        console.error('Fetch error:', error);
    });

}
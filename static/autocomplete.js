// CineFlix AI - Autocomplete Engine
// Crafted by Swastik Sengupta

new autoComplete({
    data: {
        src: films,
    },
    selector: "#autoComplete",
    threshold: 1,
    debounce: 60,
    searchEngine: "loose",
    resultsList: {
        render: true,
        container: source => {
            source.setAttribute("id", "autoComplete_list");
        },
        destination: document.querySelector(".search-box-wrapper"),
        position: "afterend",
        element: "ul"
    },
    maxResults: 6,
    highlight: true,
    resultItem: {
        content: (data, source) => {
            source.innerHTML = "<i class='fa-solid fa-film mr-2' style='color:#e50914;font-size:12px;'></i> " + data.match;
        },
        element: "li"
    },
    noResults: () => {
        const list = document.querySelector("#autoComplete_list");
        if (list) {
            list.innerHTML = "<li style='color:#64748b;padding:10px 16px;cursor:default;'>No matching movies found in database</li>";
        }
    },
    onSelection: feedback => {
        const val = feedback.selection.value;
        document.getElementById('autoComplete').value = val;
        $('.movie-button').attr('disabled', false).trigger('click');
    }
});
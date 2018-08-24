$(document).ready( function () {
    $('form').on('submit', function (event) {
        $.ajax({
            data : {
                query : $('#searchQuery').val(),
            },
            type : 'POST',
            url : '/search'
        })
        .done(display_results);
        event.preventDefault();
    });
});

function display_results(data) {
    // is_correct := if True, display alert-success else alert-danger
    var search = data.search
    if ('error' in data) {
        $("#searchResults").empty()
        $("#searchResults").append(print_error_alert())
        $(`#errorAlert`).text(data.error).show();
        return
    }

    $("#searchResults").empty()
    for (var i = 0; i < search.length; i++) {
        var clip_i = search[i]
        var j = i + 1
        $("#searchResults").append(print_search_result(j))
        // Update content of search result
        $(`#clip${j}Rank`).text(`Rank: ${j}`);
        $(`#clip${j}Name`).text(`Video: ${clip_i.video}`);
        $(`#clip${j}Segmentid`).text(`Segment: ${clip_i.segment[0]} - ${clip_i.segment[1]}`);
        $(`#clip${j}Score`).text(`Score: ${clip_i.score}`);
        $(`#clip${j}GIF`).attr("src", `${clip_i.path}`);
        // if ( clip_i.true_positive ) {
        //     $(`#Rank${j}Alert`).addClass("alert-success")
        // } else {
        //     $(`#Rank${j}Alert`).addClass("alert-danger")
        // }
    }
}

function print_search_result(index) {
    // would be better to use templates, we could place results right away
    var html_str = `<div id="Rank${index}Alert" class="alert" role="alert">`
    html_str += `<div class="row" id="rowRank${index}">`
    html_str += '<div class="col-6">'
    html_str += `<img id="clip${index}GIF" class="img-thumbnail" style="width:100%;max-width:320px;height:auto;">`
    html_str += '</div>'
    html_str += '<div class="col-6">'
    html_str += `<p id="clip${index}Rank">Rank position</p>`
    html_str += `<p id="clip${index}Name">Video name</p>`
    html_str += `<p id="clip${index}Segmentid">[segment, time]</p>`
    html_str += `<p id="clip${index}Score">score</p>`
    html_str += '</div>'
    html_str += '</div>'
    html_str += '</div>'
    return html_str
}

function print_error_alert() {
    return '<div class="row" id="errorAlert" class="alert alert-error" role="alert"></div>'
}
$(document).ready( function () {
    // Setup - add a text input to each footer cell
    $('#IndexTable tfoot th').each( function () {
        var title = $(this).text();
        console.log(title)
        if (title != "") {
            $(this).html( '<input type="text" placeholder="Search by ' + title + '" />' );
        }
    } );

    var table = $('#IndexTable').DataTable( {
        select: true
    } );

    // Search results for a given query
    table.on('select', function ( e, dt, type, indexes ) {
        if ( type === 'row' ) {
            var sentence_query = table.rows( indexes).data().pluck(0)[0]
            get_sentence_results(sentence_query)
        }
    } );

    // Apply the search over columns
    table.columns().every( function () {
        var that = this;

        $( 'input', this.footer() ).on( 'keyup change', function () {
            if ( that.search() !== this.value ) {
                that
                    .search( this.value )
                    .draw();
            }
        } );
    } );
});

function get_sentence_results(sentence) {
    $.ajax({
        data : {
            query : sentence,
        },
        type : 'POST',
        url : '/get_sentence_results'
    })
    .done(display_results);
    event.preventDefault();
}

function display_results(data) {
    // is_correct := if True, display alert-success else alert-danger
    var search = data.search
    if ('error' in data) {
        console.log('WIP: report failure')
        return
    }
    update_video(data)
    update_sentences(data)
}

function update_sentences(data) {
    $("#colSentences").empty()
    var html_str = `<div class="col">`
    html_str += `<p>Description: ${data.description}</p>`;
    for (var i = 0; i < data.topk.length; i++) {
        html_str += `<p>Rank ${i + 1}: ${data.topk[i]}</p>`;
    }
    html_str += '</div>'
    $("#colSentences").append(html_str)
}

function update_video(data) {
    $("#colVideo").empty()
    // flickr player
    html_str = '<div>'
    html_str += '<a data-flickr-embed="true" data-context="true" '
    html_str += `href="https://www.flickr.com/photos/${data.video}/in/photostream/">`
    html_str += ' <img src="https://farm4.staticflickr.com/3259/2408598493_655c93f5f9.jpg"'
    html_str += ' width="320" height="240" alt="2005_03_13__11_28_05"></a>'
    html_str += '<script async src="//embedr.flickr.com/assets/client-code.js"'
    html_str += ' charset="utf-8"></script>'
    html_str += '</div>'
    // metadata
    html_str += '<div>'
    html_str += `<p>Name: ${data.video}</p>`
    html_str += `<p>Time: ${data.time[0] * 5} - ${data.time[1] * 5 + 5}</p>`
    html_str += `<p>Under/Unseen nouns: ${data.noun_subset.join(' | ')}</p>`
    html_str += `<p>Not-targeted nouns: ${data.untargeted_nouns.join(' | ')}</p>`
    html_str += '</div>'
    $("#colVideo").append(html_str)
}

function print_error_alert() {
    return '<div class="row" id="errorAlert" class="alert alert-error" role="alert"></div>'
}
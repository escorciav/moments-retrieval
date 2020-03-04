$(document).ready( function () {
    // Setup - add a text input to each footer cell
    $('#IndexTable tfoot th').each( function () {
        var title = $(this).text();
        console.log(title)
        if (title != "") {
            $(this).html( '<input type="text" placeholder="Search '+title+'" />' );
        }
    } );

    var table = $('#IndexTable').DataTable( {
        select: true
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


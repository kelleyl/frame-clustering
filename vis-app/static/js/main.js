// place any jQuery/helper plugins in here, instead of separate, slower script files.
$(document).ready(function() {
    $('#exampleModal').on('show.bs.modal', function (event) {
        console.log("called");
        var button = $(event.relatedTarget); // Button that triggered the modal
        var recipient = button.data('whatever'); // Extract info from data-* attributes
        console.log(recipient);
        // If necessary, you could initiate an AJAX request here (and then do the updating in a callback).
        // Update the modal's content. We'll use jQuery here, but you could use a data binding library or other methods instead.
        var modal = $(this);
        modal.find('.modal-title').text('New message to ' + recipient);
        modal.find('.modal-body input').val(recipient);
        modal.find('#contents').html('<embed src="http://localhost:5000/static/img/all/28375.pdf">');

    });
});

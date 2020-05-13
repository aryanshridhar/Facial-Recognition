function readURL1(input) {

    if (input.files && input.files[0]) {

        var reader = new FileReader();
        reader.onload = function (e) {

            $('#profile-img-tag1').attr('src', e.target.result);

        }
        reader.readAsDataURL(input.files[0]);
    }
}

function readURL2(input) {

    if (input.files && input.files[0]) {

        var reader = new FileReader();
        reader.onload = function (e) {

            $('#profile-img-tag2').attr('src', e.target.result);

        }
        reader.readAsDataURL(input.files[0]);
    }
}


$("#file1").change(function(){
    readURL1(this);
});
$("#file2").change(function(){
    readURL2(this);
});

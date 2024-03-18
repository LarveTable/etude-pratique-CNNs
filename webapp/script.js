var methods = [];
var neural_network = "";
var images = [];

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.image_select').forEach(function(radio) {
        radio.addEventListener('change', function() {
            var value = this.getAttribute('value');
            if (value == 'single') {
                var single = document.querySelector('.single');
                single.style.display = 'flex';
                var multiple = document.querySelector('.multiple');
                multiple.style.display = 'none';
            }
            else{
                var multiple = document.querySelector('.multiple');
                multiple.style.display = 'flex';
                var single = document.querySelector('.single');
                single.style.display = 'none';
            }
        });
    });

    document.querySelectorAll('.XAI').forEach(function(radio) {
        radio.addEventListener('change', function() {
            var value = this.getAttribute('value');
            if (radio.checked) {
                methods.push(value);
            }
            else{
                methods.splice(methods.indexOf(value), 1);
            }
        });
    });

    document.querySelectorAll('.NN').forEach(function(radio) {
        radio.addEventListener('change', function() {
            var value = this.getAttribute('value');
            neural_network = value;
        });
    });

    document.getElementById('run').addEventListener('click', function() {
        if (methods.length == 0) {
            alert('Please select at least one XAI method');
            return;
        }
        else if (neural_network == "") {
            alert('Please select a neural network');
            return;
        }
        else if (images.length == 0) {
            alert('Please select at least one image');
            return;
        }
        else{
            console.log('Ready to run with the following parameters:'+methods+neural_network+images);
        }
    });

    document.getElementById('folder').addEventListener('change', function(event) {
        var files = event.target.files;
        images = files;
        console.log(files); // Output selected files to console for demonstration
        // Now you can do whatever you want with the selected files
    });

    document.getElementById('file').addEventListener('change', function(event) {
        var files = event.target.files;
        images = files;
        console.log(files); // Output selected files to console for demonstration
        // Now you can do whatever you want with the selected files
    });
});
(function ($) {
    'use strict';

    var $fileSelector = document.getElementById('fileUpload');
    var $imgSelector = $('img.preview');
    var $resultSelector = $('span.result');
    var $confidenceList = $(".confidences");
    var $confidenceSection = $(".confidence-objects");

    function addConfidence(color, label, confidence) {

        var element = $('<div class="confidence" style="box-shadow: inset 2em 0 0 #'+ color +'"><span class="item">'+ label + " " + confidence.toFixed(2) +'</span></div>');
        $confidenceList.append(element);
    }

    function drawBoundingBoxes(boundingBoxes) {
        $confidenceSection.show();
        $confidenceList.empty();
        var c = document.getElementById("myCanvas");
        var ctx = c.getContext("2d");
        ctx.clearRect(0, 0, c.width, c.height);
        var img = document.getElementById("orgimg");
        ctx.drawImage(img, 0, 0);

        for (var i = 0; i < boundingBoxes.length; i++) {
            var randomColor = Math.floor(Math.random() * 16777215).toString(16);
            addConfidence(randomColor, boundingBoxes[i].Label, boundingBoxes[i].Confidence);
            ctx.beginPath();
            ctx.strokeStyle = "#" + randomColor;
            ctx.lineWidth = 4;
            ctx.strokeRect(boundingBoxes[i].X, boundingBoxes[i].Y, boundingBoxes[i].Width, boundingBoxes[i].Height);
            ctx.closePath();
        }
    }

    function previewFile() {

        var file = $fileSelector.files[0];
        var reader = new FileReader();

        reader.onloadend = function () {
            $imgSelector.show();
            $imgSelector.attr('src', reader.result);
        };

        if (file) {
            reader.readAsDataURL(file);
        } else {
            $imgSelector.attr('src', '');
        }
    }

    $fileSelector.addEventListener('change', function () {
        previewFile();
    });

    $("a.score").on('click', function (event) {
        event.preventDefault();
        var dataImage = {
            EncodedImage: $imgSelector.attr('src').split(',').pop()
        };
        $.ajax({
            type: "POST",
            contentType: 'application/json',
            url: 'api/predictive/',
            data: JSON.stringify(dataImage),
            success: function (data) {
                drawBoundingBoxes(data);
            }
        });
    });

})($);
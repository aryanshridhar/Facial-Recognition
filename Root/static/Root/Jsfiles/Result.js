var img1 = document.getElementById('img1');
var img2 = document.getElementById('img2');

(function SetSource(){
    img1.src = sessionStorage.getItem('img1');
    img2.src = sessionStorage.getItem('img2');
})();
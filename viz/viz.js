var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
var particles = [];
var num_particles = 1000;//Change that to your liking

//Helper function to get a random color - but not too dark
function GetRandomColor() {
    var r = 0, g = 0, b = 0;
    while (r < 100 && g < 100 && b < 100)
    {
        r = Math.floor(Math.random() * 256);
        g = Math.floor(Math.random() * 256);
        b = Math.floor(Math.random() * 256);
    }

    return "rgb(" + r + "," + g + ","  + b + ")";
}
//Particle object with random starting position, velocity and color
var Particle = function () {
    this.x = canvas.width * Math.random();
    this.y = canvas.height * Math.random();
    this.vx = 4 * Math.random() - 2;
    this.vy = 4 * Math.random() - 2;
    this.Color = GetRandomColor();
}
//Ading two methods
Particle.prototype.Draw = function (ctx) {
    ctx.fillStyle = this.Color;
    ctx.fillRect(this.x, this.y, 2, 2);
}
Particle.prototype.Update = function () {
    this.x += this.vx;
    this.y += this.vy;
 
    if (this.x<0 || this.x > canvas.width)
        this.vx = -this.vx;
 
    if (this.y < 0 || this.y > canvas.height)
        this.vy = -this.vy;
}
function loop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (var i = 0; i < num_particles; i++) {
        particles[i].Update();
        particles[i].Draw(ctx);
    }
    requestAnimationFrame(loop);
}
//Create particles
for (var i = 0; i < num_particles; i++)
    particles.push(new Particle());
loop();

"""
EleGuard Browser Audio Synthesizer
Generates real-time audio sirens using HTML5 Web Audio API.
"""


def play_browser_alarm_script() -> str:
    """Return JavaScript to synthesize an emergency alarm chime in browser."""
    return """
    <script>
    (function() {
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.35);
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.35);
        } catch(e) {
            console.log("Audio alert blocked by browser autoplay policy");
        }
    })();
    </script>
    """

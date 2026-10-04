/**
 * AudioPreviewPlayer — Reusable HTML5 Audio Preview Component Controller
 * SonicSentinel AI / AcousticX Intelligence
 *
 * Provides custom playback controls (Play, Pause, Replay, Seek bar, Volume/Mute,
 * Playback Speed) for native HTML5 <audio> elements, with automatic binding to
 * file inputs (<input type="file">) or pre-loaded audio URLs.
 */

class AudioPreviewPlayer {
  constructor(config = {}) {
    this.idSuffix = config.idSuffix || 'default';
    this.fileInputId = config.fileInputId || null;
    this.initialUrl = config.audioUrl || null;
    this.initialFilename = config.filename || null;
    this.objectUrl = null;
    this.isSeeking = false;

    // DOM Elements
    this.container = document.getElementById(`audio-preview-container-${this.idSuffix}`);
    this.audioEl = document.getElementById(`audio-elem-${this.idSuffix}`);
    this.playBtn = document.getElementById(`play-btn-${this.idSuffix}`);
    this.playIcon = document.getElementById(`play-icon-${this.idSuffix}`);
    this.pauseIcon = document.getElementById(`pause-icon-${this.idSuffix}`);
    this.replayBtn = document.getElementById(`replay-btn-${this.idSuffix}`);
    this.seekBar = document.getElementById(`seek-bar-${this.idSuffix}`);
    this.currentTimeEl = document.getElementById(`current-time-${this.idSuffix}`);
    this.durationEl = document.getElementById(`duration-${this.idSuffix}`);
    this.muteBtn = document.getElementById(`mute-btn-${this.idSuffix}`);
    this.volumeBar = document.getElementById(`volume-bar-${this.idSuffix}`);
    this.volIconHigh = document.getElementById(`vol-icon-high-${this.idSuffix}`);
    this.volIconLow = document.getElementById(`vol-icon-low-${this.idSuffix}`);
    this.volIconMute = document.getElementById(`vol-icon-mute-${this.idSuffix}`);
    this.speedSelect = document.getElementById(`speed-select-${this.idSuffix}`);
    this.filenameEl = document.getElementById(`filename-${this.idSuffix}`);
    this.statusBadge = document.getElementById(`status-badge-${this.idSuffix}`);

    this.init();
  }

  init() {
    if (!this.audioEl) {
      console.warn(`[AudioPreviewPlayer] Audio element #audio-elem-${this.idSuffix} not found.`);
      return;
    }

    // Bind Play/Pause
    if (this.playBtn) {
      this.playBtn.addEventListener('click', () => this.togglePlay());
    }

    // Bind Replay
    if (this.replayBtn) {
      this.replayBtn.addEventListener('click', () => this.replay());
    }

    // Bind Seek bar
    if (this.seekBar) {
      this.seekBar.addEventListener('input', () => {
        this.isSeeking = true;
        const curSec = (parseFloat(this.seekBar.value) / 100) * (this.audioEl.duration || 0);
        if (this.currentTimeEl) {
          this.currentTimeEl.textContent = this.formatTime(curSec);
        }
        this.updateSliderGradient(this.seekBar, '#06b6d4');
      });

      this.seekBar.addEventListener('change', () => {
        if (this.audioEl.duration && !isNaN(this.audioEl.duration)) {
          this.audioEl.currentTime = (parseFloat(this.seekBar.value) / 100) * this.audioEl.duration;
        }
        this.isSeeking = false;
      });
    }

    // Bind Mute/Volume controls
    if (this.muteBtn) {
      this.muteBtn.addEventListener('click', () => this.toggleMute());
    }

    if (this.volumeBar) {
      this.volumeBar.addEventListener('input', () => {
        const val = parseFloat(this.volumeBar.value);
        this.audioEl.volume = val;
        this.audioEl.muted = (val === 0);
        this.updateVolumeIcons();
        this.updateSliderGradient(this.volumeBar, '#06b6d4');
      });
    }

    // Bind Playback Speed selector
    if (this.speedSelect) {
      this.speedSelect.addEventListener('change', () => {
        this.audioEl.playbackRate = parseFloat(this.speedSelect.value);
      });
    }

    // Native HTML5 Audio Event Listeners
    this.audioEl.addEventListener('loadedmetadata', () => this.onMetadataLoaded());
    this.audioEl.addEventListener('timeupdate', () => this.onTimeUpdate());
    this.audioEl.addEventListener('play', () => this.onPlayStateChange(true));
    this.audioEl.addEventListener('pause', () => this.onPlayStateChange(false));
    this.audioEl.addEventListener('ended', () => this.onEnded());
    this.audioEl.addEventListener('error', (e) => this.onError(e));

    // File input change listener for dynamic previewing
    if (this.fileInputId) {
      const fileInput = document.getElementById(this.fileInputId);
      if (fileInput) {
        fileInput.addEventListener('change', (e) => {
          if (e.target.files && e.target.files[0]) {
            this.loadFromFile(e.target.files[0]);
          }
        });

        // Auto-load if file was already selected before script execution
        if (fileInput.files && fileInput.files[0]) {
          this.loadFromFile(fileInput.files[0]);
        }
      }
    }

    // Load initial URL if provided
    if (this.initialUrl) {
      this.loadFromUrl(this.initialUrl, this.initialFilename);
    }

    // Initialize track fill gradients
    if (this.seekBar) this.updateSliderGradient(this.seekBar, '#06b6d4');
    if (this.volumeBar) this.updateSliderGradient(this.volumeBar, '#06b6d4');
  }

  togglePlay() {
    if (!this.audioEl.src || this.audioEl.src === 'about:blank' || this.audioEl.src === window.location.href) {
      return;
    }
    if (this.audioEl.paused) {
      this.audioEl.play().catch(err => {
        console.error('[AudioPreviewPlayer] Error initiating play:', err);
      });
    } else {
      this.audioEl.pause();
    }
  }

  replay() {
    if (!this.audioEl.src) return;
    this.audioEl.currentTime = 0;
    if (this.seekBar) {
      this.seekBar.value = 0;
      this.updateSliderGradient(this.seekBar, '#06b6d4');
    }
    this.audioEl.play().catch(err => {
      console.error('[AudioPreviewPlayer] Error replaying audio:', err);
    });
  }

  toggleMute() {
    this.audioEl.muted = !this.audioEl.muted;
    if (this.volumeBar) {
      this.volumeBar.value = this.audioEl.muted ? 0 : (this.audioEl.volume || 1);
      this.updateSliderGradient(this.volumeBar, '#06b6d4');
    }
    this.updateVolumeIcons();
  }

  loadFromFile(file) {
    if (!file) return;

    if (this.objectUrl) {
      URL.revokeObjectURL(this.objectUrl);
      this.objectUrl = null;
    }

    this.objectUrl = URL.createObjectURL(file);
    this.audioEl.src = this.objectUrl;

    if (this.filenameEl) {
      this.filenameEl.textContent = file.name;
    }

    if (this.statusBadge) {
      this.statusBadge.textContent = 'Ready';
      this.statusBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/30';
    }

    if (this.container) {
      this.container.classList.remove('hidden');
    }
  }

  loadFromUrl(url, filename = null) {
    if (!url) return;
    this.audioEl.src = url;
    if (filename && this.filenameEl) {
      this.filenameEl.textContent = filename;
    }
    if (this.statusBadge) {
      this.statusBadge.textContent = 'Ready';
      this.statusBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/30';
    }
    if (this.container) {
      this.container.classList.remove('hidden');
    }
  }

  onMetadataLoaded() {
    const dur = this.audioEl.duration;
    if (this.durationEl && !isNaN(dur)) {
      this.durationEl.textContent = this.formatTime(dur);
    }
    if (this.currentTimeEl) {
      this.currentTimeEl.textContent = this.formatTime(this.audioEl.currentTime);
    }
    if (this.playBtn) {
      this.playBtn.disabled = false;
    }
  }

  onTimeUpdate() {
    if (this.isSeeking) return;

    const dur = this.audioEl.duration;
    const cur = this.audioEl.currentTime;

    if (this.currentTimeEl) {
      this.currentTimeEl.textContent = this.formatTime(cur);
    }

    if (dur && !isNaN(dur) && dur > 0) {
      const pct = (cur / dur) * 100;
      if (this.seekBar) {
        this.seekBar.value = pct;
        this.updateSliderGradient(this.seekBar, '#06b6d4');
      }
    }
  }

  onPlayStateChange(isPlaying) {
    if (isPlaying) {
      if (this.playIcon) this.playIcon.classList.add('hidden');
      if (this.pauseIcon) this.pauseIcon.classList.remove('hidden');
      if (this.statusBadge) {
        this.statusBadge.textContent = 'Playing';
        this.statusBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 animate-pulse';
      }
    } else {
      if (this.playIcon) this.playIcon.classList.remove('hidden');
      if (this.pauseIcon) this.pauseIcon.classList.add('hidden');
      if (this.statusBadge) {
        this.statusBadge.textContent = 'Paused';
        this.statusBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide uppercase bg-amber-500/20 text-amber-300 border border-amber-500/30';
      }
    }
  }

  onEnded() {
    if (this.playIcon) this.playIcon.classList.remove('hidden');
    if (this.pauseIcon) this.pauseIcon.classList.add('hidden');
    if (this.seekBar) {
      this.seekBar.value = 0;
      this.updateSliderGradient(this.seekBar, '#06b6d4');
    }
    if (this.currentTimeEl) {
      this.currentTimeEl.textContent = this.formatTime(0);
    }
    if (this.statusBadge) {
      this.statusBadge.textContent = 'Ended';
      this.statusBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide uppercase bg-slate-800 text-slate-400 border border-slate-700';
    }
  }

  onError(err) {
    console.error('[AudioPreviewPlayer] HTML5 Audio error event:', err);
    if (this.statusBadge) {
      this.statusBadge.textContent = 'Error';
      this.statusBadge.className = 'px-2.5 py-0.5 rounded-full text-[10px] font-semibold tracking-wide uppercase bg-rose-500/20 text-rose-300 border border-rose-500/30';
    }
  }

  updateVolumeIcons() {
    const isMuted = this.audioEl.muted || this.audioEl.volume === 0;
    const vol = this.audioEl.volume;

    if (isMuted) {
      if (this.volIconMute) this.volIconMute.classList.remove('hidden');
      if (this.volIconHigh) this.volIconHigh.classList.add('hidden');
      if (this.volIconLow) this.volIconLow.classList.add('hidden');
    } else if (vol < 0.5) {
      if (this.volIconLow) this.volIconLow.classList.remove('hidden');
      if (this.volIconHigh) this.volIconHigh.classList.add('hidden');
      if (this.volIconMute) this.volIconMute.classList.add('hidden');
    } else {
      if (this.volIconHigh) this.volIconHigh.classList.remove('hidden');
      if (this.volIconLow) this.volIconLow.classList.add('hidden');
      if (this.volIconMute) this.volIconMute.classList.add('hidden');
    }
  }

  updateSliderGradient(inputEl, activeColor = '#06b6d4') {
    if (!inputEl) return;
    const min = parseFloat(inputEl.min) || 0;
    const max = parseFloat(inputEl.max) || 100;
    const val = parseFloat(inputEl.value) || 0;
    const pct = ((val - min) / (max - min)) * 100;
    inputEl.style.background = `linear-gradient(to right, ${activeColor} 0%, ${activeColor} ${pct}%, #334155 ${pct}%, #334155 100%)`;
  }

  formatTime(seconds) {
    if (isNaN(seconds) || !isFinite(seconds) || seconds < 0) {
      return '00:00';
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const padMins = mins < 10 ? `0${mins}` : `${mins}`;
    const padSecs = secs < 10 ? `0${secs}` : `${secs}`;
    return `${padMins}:${padSecs}`;
  }
}

// Global initialization helper
window.initAudioPreview = function (config) {
  return new AudioPreviewPlayer(config);
};

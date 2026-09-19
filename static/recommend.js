/**
 * CineFlix AI - Frontend Interaction & Async Engine
 * Crafted by Swastik Sengupta
 */

$(function() {
  const source = document.getElementById('autoComplete');

  // Enable/disable search button dynamically
  const inputHandler = function(e) {
    if (!e.target.value.trim()) {
      $('.movie-button').attr('disabled', true);
    } else {
      $('.movie-button').attr('disabled', false);
    }
  };

  if (source) {
    source.addEventListener('input', inputHandler);

    // Support pressing Enter inside the search field
    source.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        if (source.value.trim()) {
          $('.movie-button').trigger('click');
        }
      }
    });
  }

  // Handle Find Movies click
  $('.movie-button').on('click', function() {
    const title = $('.movie').val().trim();
    if (!title) {
      $('.fail').slideDown();
      $('.results').empty();
      return;
    }
    loadMovieDetails(title);
  });
});

// Invoked when user clicks any of the 10 recommended movie cards
function recommendcard(el) {
  const title = $(el).attr('title');
  if (title) {
    $('#autoComplete').val(title);
    $('.movie-button').attr('disabled', false);
    loadMovieDetails(title);
  }
}

// Master Async Orchestration Flow
async function loadMovieDetails(title) {
  $("#loader").fadeIn(200);
  $('.fail').slideUp(200);

  try {
    // 1. Search movie in TMDB via backend proxy
    const searchRes = await fetch('/api/tmdb/search?query=' + encodeURIComponent(title));
    const searchData = await searchRes.json();

    if (!searchData.results || searchData.results.length === 0) {
      showError();
      return;
    }

    // Match exact title first, then prefix/word match, before falling back to results[0]
    const cleanQuery = title.trim().toLowerCase();
    let movie = searchData.results.find(m => 
      (m.title && m.title.trim().toLowerCase() === cleanQuery) ||
      (m.original_title && m.original_title.trim().toLowerCase() === cleanQuery)
    );

    if (!movie) {
      movie = searchData.results.find(m => 
        (m.title && (m.title.trim().toLowerCase().startsWith(cleanQuery) || cleanQuery.startsWith(m.title.trim().toLowerCase()))) ||
        (m.original_title && (m.original_title.trim().toLowerCase().startsWith(cleanQuery) || cleanQuery.startsWith(m.original_title.trim().toLowerCase())))
      ) || searchData.results[0];
    }

    const movieId = movie.id;
    const movieTitle = movie.title || movie.original_title;

    // 2. Fetch Cosine Similarity Recommendations from Flask
    const simFormData = new FormData();
    simFormData.append('name', movieTitle);

    let simRes = await fetch('/similarity', {
      method: 'POST',
      body: simFormData
    });
    let simText = await simRes.text();

    // Fallback: If TMDB title didn't match our dataset, try the original query string
    if (simText.startsWith("Sorry!") && movieTitle.toLowerCase() !== title.toLowerCase()) {
      const fallbackForm = new FormData();
      fallbackForm.append('name', title);
      const fallbackRes = await fetch('/similarity', { method: 'POST', body: fallbackForm });
      const fallbackText = await fallbackRes.text();
      if (!fallbackText.startsWith("Sorry!")) {
        simText = fallbackText;
      }
    }

    if (simText.startsWith("Sorry!")) {
      showError();
      return;
    }

    const recMovieTitles = simText.split('---').map(t => t.trim()).filter(Boolean);

    // 3. Fetch Full Movie Details & Cast Credits in parallel
    const [detailsRes, creditsRes] = await Promise.all([
      fetch('/api/tmdb/movie/' + movieId),
      fetch('/api/tmdb/credits/' + movieId)
    ]);

    const movieDetails = await detailsRes.json();
    const movieCredits = await creditsRes.json();

    // 4. Fetch Posters for Recommended Movies concurrently
    const recPostersPromises = recMovieTitles.map(async (recTitle) => {
      try {
        const r = await fetch('/api/tmdb/search?query=' + encodeURIComponent(recTitle));
        const d = await r.json();
        if (d.results && d.results.length > 0 && d.results[0].poster_path) {
          return 'https://image.tmdb.org/t/p/w500' + d.results[0].poster_path;
        }
      } catch (err) {
        console.warn('Poster fetch error for', recTitle, err);
      }
      return '/static/image.jpg';
    });

    const recPosters = await Promise.all(recPostersPromises);

    // 5. Extract top 8 cast members and fetch their individual bio details
    const topCast = (movieCredits.cast || []).slice(0, 8);
    const castIds = [];
    const castNames = [];
    const castChars = [];
    const castProfiles = [];

    topCast.forEach(c => {
      castIds.push(c.id);
      castNames.push(c.name);
      castChars.push(c.character || 'Cast');
      castProfiles.push(c.profile_path ? ('https://image.tmdb.org/t/p/w300' + c.profile_path) : '/static/image.jpg');
    });

    const castBioPromises = castIds.map(async (personId) => {
      try {
        const pRes = await fetch('/api/tmdb/person/' + personId);
        const pData = await pRes.json();
        return {
          birthday: pData.birthday ? new Date(pData.birthday).toDateString().split(' ').slice(1).join(' ') : 'N/A',
          place: pData.place_of_birth || 'N/A',
          bio: pData.biography || 'No biography available for this artist.'
        };
      } catch (e) {
        return { birthday: 'N/A', place: 'N/A', bio: 'No biography available.' };
      }
    });

    const castBiosData = await Promise.all(castBioPromises);
    const castBdays = castBiosData.map(b => b.birthday);
    const castBios = castBiosData.map(b => b.bio);
    const castPlaces = castBiosData.map(b => b.place);

    // 6. Format Movie Metadata
    const genresList = (movieDetails.genres || []).map(g => g.name).join(', ');
    const runtimeMins = parseInt(movieDetails.runtime) || 0;
    const runtimeFormatted = runtimeMins > 0 
      ? (Math.floor(runtimeMins / 60) + 'h ' + (runtimeMins % 60) + 'm')
      : 'N/A';
    const releaseDateFormatted = movieDetails.release_date
      ? new Date(movieDetails.release_date).toDateString().split(' ').slice(1).join(' ')
      : 'Unknown';
    const posterUrl = movieDetails.poster_path 
      ? ('https://image.tmdb.org/t/p/w500' + movieDetails.poster_path) 
      : '/static/image.jpg';

    // 7. POST to /recommend for NLP Sentiment Classification & Final Rendering
    const finalFormData = new FormData();
    finalFormData.append('title', movieTitle);
    finalFormData.append('movie_id', movieId);
    finalFormData.append('imdb_id', movieDetails.imdb_id || '');
    finalFormData.append('poster', posterUrl);
    finalFormData.append('genres', genresList);
    finalFormData.append('overview', movieDetails.overview || 'No synopsis available.');
    finalFormData.append('rating', movieDetails.vote_average ? movieDetails.vote_average.toFixed(1) : 'N/A');
    finalFormData.append('vote_count', (movieDetails.vote_count || 0).toLocaleString());
    finalFormData.append('release_date', releaseDateFormatted);
    finalFormData.append('runtime', runtimeFormatted);
    finalFormData.append('status', movieDetails.status || 'Released');
    finalFormData.append('rec_movies', JSON.stringify(recMovieTitles));
    finalFormData.append('rec_posters', JSON.stringify(recPosters));
    finalFormData.append('cast_ids', JSON.stringify(castIds));
    finalFormData.append('cast_names', JSON.stringify(castNames));
    finalFormData.append('cast_chars', JSON.stringify(castChars));
    finalFormData.append('cast_profiles', JSON.stringify(castProfiles));
    finalFormData.append('cast_bdays', JSON.stringify(castBdays));
    finalFormData.append('cast_bios', JSON.stringify(castBios));
    finalFormData.append('cast_places', JSON.stringify(castPlaces));

    const recHtmlResponse = await fetch('/recommend', {
      method: 'POST',
      body: finalFormData
    });

    if (!recHtmlResponse.ok) {
      throw new Error('Recommend API returned status ' + recHtmlResponse.status);
    }

    const htmlContent = await recHtmlResponse.text();

    // 8. Inject and display with smooth scroll
    $('.results').html(htmlContent);
    $('.fail').slideUp();
    $("#loader").fadeOut(250);

    $('html, body').animate({
      scrollTop: $('.results').offset().top - 80
    }, 500);

  } catch (error) {
    console.error('Movie recommendation pipeline error:', error);
    showError();
  }
}

function showError() {
  $("#loader").fadeOut(200);
  $('.results').empty();
  $('.fail').slideDown(200);
}

// VisionOS Spatial Modal Handlers
function openVisionModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
}

function closeVisionModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.remove('active');
  }
  // Restore body scroll if no modal active
  if (!document.querySelector('.vision-modal-backdrop.active')) {
    document.body.style.overflow = '';
  }
}

// Global listener for Escape key to close modal
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.querySelectorAll('.vision-modal-backdrop.active').forEach(function(m) {
      m.classList.remove('active');
    });
    document.body.style.overflow = '';
  }
});


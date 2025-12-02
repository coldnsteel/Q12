// q12_report_template.js - Dynamically loads and renders Q12 report from JSON

fetch('latest_report.json')
  .then(response => {
    if (!response.ok) {
      throw new Error('Failed to load report data');
    }
    return response.json();
  })
  .then(data => {
    const container = document.getElementById('report-container');
    data.forEach(item => {
      const card = document.createElement('div');
      card.className = 'card p-6 rounded-xl mb-6 shadow-xl';

      // Header
      const header = document.createElement('div');
      header.className = 'flex flex-col md:flex-row justify-between items-start md:items-center border-b border-gray-700 pb-4 mb-4';
      header.innerHTML = `
        <h3 class="text-4xl font-extrabold text-white tracking-wide">${item.ticker}</h3>
        <div class="mt-2 md:mt-0 flex items-center space-x-3">
          <span class="text-lg font-medium text-gray-300">TTE: ${item.tte_weeks.toFixed(1)} Weeks</span>
          <span class="score-${item.score >= 70 ? '70' : '50'} px-4 py-1 rounded-full text-xl font-bold ${item.score >= 70 ? 'pulse-strong' : ''}">
            SCORE ${item.score}
          </span>
        </div>
      `;
      card.appendChild(header);

      // Classification
      const classification = document.createElement('p');
      classification.className = `text-2xl font-semibold ${item.score >= 70 ? 'text-green-400' : 'text-yellow-300'} mb-4`;
      classification.textContent = `Classification: ${item.classification}`;
      card.appendChild(classification);

      // Signal Breakdown Grid
      const grid = document.createElement('div');
      grid.className = 'grid grid-cols-1 md:grid-cols-2 gap-4';

      // Behavioral Signals
      const behavioral = document.createElement('div');
      behavioral.className = 'bg-gray-800 p-4 rounded-lg';
      behavioral.innerHTML = `
        <p class="font-bold text-white mb-2">Behavioral Signals (Weighted & Time-Decayed)</p>
        <ul class="text-gray-300 text-sm space-y-1">
          <li class="flex justify-between"><span>Incremental Accumulation:</span> <span class="text-green-300">Score ${item.signals.accum.score} (${item.signals.accum.flags.join(', ')})</span></li>
          <li class="flex justify-between"><span>Short Interest Drop:</span> <span class="text-green-300">Score ${item.signals.short.score} (${item.signals.short.flags.join(', ')})</span></li>
          <li class="flex justify-between"><span>ML Anomaly Bonus:</span> <span class="text-yellow-300">+${item.ml_bonus} Bonus</span></li>
        </ul>
      `;
      grid.appendChild(behavioral);

      // Fundamental Signals
      const fundamental = document.createElement('div');
      fundamental.className = 'bg-gray-800 p-4 rounded-lg';
      fundamental.innerHTML = `
        <p class="font-bold text-white mb-2">Fundamental & Insider Signals</p>
        <ul class="text-gray-300 text-sm space-y-1">
          <li class="flex justify-between"><span>Insider Buys (Form 4):</span> <span class="text-green-300">Score ${item.signals.insider.score} (${item.signals.insider.flags.join(', ')})</span></li>
          <li class="flex justify-between"><span>13F Institutional Change:</span> <span class="text-green-300">Score ${item.signals['13f'].score} (${item.signals['13f'].flags.join(', ')})</span></li>
          <li class="flex justify-between"><span>News Proxy (Contracts/Govt):</span> <span class="text-yellow-300">Score ${item.signals.news.score} (${item.signals.news.flags.join(', ')})</span></li>
        </ul>
      `;
      grid.appendChild(fundamental);

      card.appendChild(grid);

      // Visualization Placeholder (assuming charts are saved as images; could load dynamically if paths in JSON)
      const viz = document.createElement('div');
      viz.className = 'mt-6';
      viz.innerHTML = `
        <p class="text-lg font-medium text-white border-t border-gray-700 pt-4">Visualization</p>
        <div class="h-32 bg-gray-700 mt-2 rounded-lg flex items-center justify-center text-gray-400">
          [Chart: q12_reports/${item.ticker}_volume.png]
        </div>
      `;
      card.appendChild(viz);

      container.appendChild(card);
    });
  })
  .catch(error => {
    console.error('Error loading report:', error);
    const container = document.getElementById('report-container');
    container.innerHTML = '<p class="text-red-400">Error loading report data. Please run q12_agent.py first.</p>';
  });

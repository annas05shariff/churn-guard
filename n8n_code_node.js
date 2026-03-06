const data = $input.first().json;

const high   = data.high_risk_count   || 0;
const medium = data.medium_risk_count || 0;
const low    = data.low_risk_count    || 0;
const total  = data.total_scanned     || 0;
const customers = data.customers || [];

const today = new Date().toLocaleDateString('en-US', { weekday:'long', month:'long', day:'numeric', year:'numeric' });
const time  = new Date().toLocaleTimeString('en-US', { hour:'2-digit', minute:'2-digit' });

// Escape special chars for Telegram MarkdownV2
function esc(str) {
  return String(str).replace(/[_*[\]()~`>#+\-=|{}.!]/g, '\\$&');
}

const top5 = customers.filter(c => c.risk_level === 'High').slice(0, 5);
const rows = top5.map(c => {
  const prob = (c.churn_probability * 100).toFixed(1);
  return `• \`${c.customer_id}\` — *${esc(prob)}%* — ${esc(c.top_driver)}`;
}).join('\n');

const text = `🛡 *ChurnGuard Daily Digest*
📅 ${esc(today)} \\| ${esc(time)}

📊 *Risk Summary*
🔴 High Risk: *${esc(high)}*
🟡 Medium Risk: *${esc(medium)}*
🟢 Low Risk: *${esc(low)}*
📋 Total Scanned: *${esc(total.toLocaleString())}*

🔴 *Top High\\-Risk Customers*
${rows}

_View dashboard at localhost:8000/dashboard_`;

return [{ json: { text } }];

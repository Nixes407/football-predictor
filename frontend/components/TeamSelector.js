/**
 * TeamSelector — a labelled dropdown of team names.
 *
 * Props:
 *   label    string   e.g. "Home Team"
 *   teams    string[] all team names for the selected league
 *   value    string   currently selected team
 *   onChange function called with the newly selected team name
 *   exclude  string   team name to hide (the team already picked on the other side)
 */
export default function TeamSelector({ label, teams, value, onChange, exclude }) {
  const available = (teams || []).filter((team) => team !== exclude);
  const isDisabled = available.length === 0;

  return (
    <div className="mb-4">
      <label className="mb-1 block text-sm font-medium text-gray-700">
        {label}
      </label>
      <select
        className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:cursor-not-allowed disabled:bg-gray-100 disabled:text-gray-400"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={isDisabled}
      >
        <option value="">Select team...</option>
        {available.map((team) => (
          <option key={team} value={team}>
            {team}
          </option>
        ))}
      </select>
    </div>
  );
}

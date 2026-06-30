import { defineCommand } from "citty";
import color from "picocolors";
import { installSkill, listSkills, removeSkill, setSkillEnabled } from "@chimpoe/core";
import { resolveAgent } from "../utils/resolve.ts";

export default defineCommand({
  meta: {
    name: "skills",
    description: "Manage agent skills (install, list, remove, toggle)",
  },
  args: {
    action: {
      type: "positional",
      required: false,
      description: "list | install | remove | enable | disable",
    },
    agent: {
      type: "positional",
      required: false,
      description: "Agent id/name (for list/toggle) or name (for install)",
    },
    name: {
      type: "string",
      description: "Skill name (for install/remove/toggle)",
    },
    body: {
      type: "string",
      description: "Skill body markdown (for install)",
    },
    file: {
      type: "string",
      description: "Path to a markdown file to read skill body from",
    },
  },
  run: async ({ args }) => {
    const action = args.action ?? "list";

    if (action === "list") {
      const agents = await import("@chimpoe/core").then((m) => m.listAgents());
      if (!args.agent) {
        console.error(color.red("Specify an agent. Usage: chimpoe skills list <agent>"));
        process.exit(2);
      }
      const me = resolveAgent(agents, args.agent);
      if (!me) {
        console.error(color.red(`No agent matching "${args.agent}".`));
        process.exit(1);
      }
      const skills = await listSkills(me.id);
      if (skills.length === 0) {
        console.log(color.gray(`No skills installed for ${me.name}.`));
        return;
      }
      for (const s of skills) {
        const state = s.enabled ? color.green("●") : color.gray("○");
        const tags = s.tags.length > 0 ? color.dim(` [${s.tags.join(", ")}]`) : "";
        console.log(
          `${state} ${s.name.padEnd(28)} v${s.version.padEnd(8)} ${s.description}${tags}`,
        );
      }
      console.log(color.dim(`\n  ${skills.length} skill(s) total`));
      return;
    }

    if (action === "install") {
      const agents = await import("@chimpoe/core").then((m) => m.listAgents());
      if (!args.agent) {
        console.error(color.red("Usage: chimpoe skills install <agent> --name X --body Y"));
        process.exit(2);
      }
      const me = resolveAgent(agents, args.agent);
      if (!me) {
        console.error(color.red(`No agent matching "${args.agent}".`));
        process.exit(1);
      }
      if (!args.name) {
        console.error(color.red("--name is required for install"));
        process.exit(2);
      }
      let body = args.body;
      if (!body && args.file) {
        try {
          body = await Bun.file(args.file).text();
        } catch {
          console.error(color.red(`Cannot read file: ${args.file}`));
          process.exit(1);
        }
      }
      if (!body) {
        console.error(color.red("--body or --file required"));
        process.exit(2);
      }
      const skill = await installSkill({
        agentId: me.id,
        name: args.name,
        body,
      });
      console.log(
        color.green("●") + ` installed skill ${color.cyan(skill.name)} for ${color.cyan(me.name)}`,
      );
      return;
    }

    if (action === "remove") {
      const agents = await import("@chimpoe/core").then((m) => m.listAgents());
      const me = resolveAgent(agents, args.agent ?? "");
      if (!me || !args.name) {
        console.error(color.red("Usage: chimpoe skills remove <agent> --name X"));
        process.exit(2);
      }
      const result = await removeSkill(me.id, args.name);
      if (result.removed) {
        console.log(color.green("●") + ` removed ${color.cyan(args.name)}`);
      } else {
        console.error(color.red(`Skill "${args.name}" not found.`));
        process.exit(1);
      }
      return;
    }

    if (action === "enable" || action === "disable") {
      const agents = await import("@chimpoe/core").then((m) => m.listAgents());
      const me = resolveAgent(agents, args.agent ?? "");
      if (!me || !args.name) {
        console.error(color.red(`Usage: chimpoe skills ${action} <agent> --name X`));
        process.exit(2);
      }
      const result = await setSkillEnabled(me.id, args.name, action === "enable");
      console.log(result.message);
      return;
    }

    console.error(color.red(`Unknown action. Use list, install, remove, enable, disable.`));
    process.exit(2);
  },
});

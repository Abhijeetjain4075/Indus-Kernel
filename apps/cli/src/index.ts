#!/usr/bin/env node
/**
 * Indus Kernel CLI — the official command-line interface.
 *
 * Subcommands will include: init, dev, deploy, agents, memory, run, status, logs.
 */
import { Command } from "commander";

const program = new Command();

program
  .name("ik")
  .description("Indus Kernel CLI")
  .version("0.1.0");

program
  .command("hello")
  .description("Print a greeting from Indus Kernel")
  .action(() => {
    console.log("Hello from Indus Kernel CLI! v0.1.0");
  });

program.parse();

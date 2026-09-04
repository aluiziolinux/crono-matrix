#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define PATH_CAP 4096
#define LINE_CAP 512
#define MIB_BYTES 1048576LL

static volatile sig_atomic_t keep_running = 1;

static void on_signal(int signo) {
    (void) signo;
    keep_running = 0;
}

static long read_mem_available_mb(void) {
    FILE *file = fopen("/proc/meminfo", "re");
    char line[LINE_CAP];
    long kib = -1;
    if (file == NULL) return -1;
    while (fgets(line, sizeof(line), file) != NULL) {
        if (sscanf(line, "MemAvailable: %ld kB", &kib) == 1) break;
    }
    fclose(file);
    return kib >= 0 ? kib / 1024 : -1;
}

static int resolve_cgroup_dir(pid_t pid, char *output, size_t capacity) {
    char proc_path[128];
    char line[PATH_CAP];
    FILE *file;
    snprintf(proc_path, sizeof(proc_path), "/proc/%ld/cgroup", (long) pid);
    file = fopen(proc_path, "re");
    if (file == NULL) return -1;
    while (fgets(line, sizeof(line), file) != NULL) {
        char *relative;
        if (strncmp(line, "0::", 3) != 0) continue;
        relative = line + 3;
        relative[strcspn(relative, "\r\n")] = '\0';
        if (strstr(relative, "..") != NULL) {
            fclose(file);
            errno = EINVAL;
            return -1;
        }
        if (strstr(relative, "/crono-llama-") == NULL
                || strstr(relative, ".scope") == NULL) {
            fclose(file);
            errno = EPERM;
            return -1;
        }
        if (snprintf(output, capacity, "/sys/fs/cgroup%s", relative)
                >= (int) capacity) {
            fclose(file);
            errno = ENAMETOOLONG;
            return -1;
        }
        fclose(file);
        return 0;
    }
    fclose(file);
    errno = ENOENT;
    return -1;
}

static int make_control_path(
        const char *directory, const char *name, char *output, size_t capacity) {
    if (snprintf(output, capacity, "%s/%s", directory, name) >= (int) capacity) {
        errno = ENAMETOOLONG;
        return -1;
    }
    return 0;
}

/* Return -2 for "max", -1 on error, otherwise MiB rounded up. */
static long read_cgroup_mb(const char *path) {
    char value[96];
    char *end = NULL;
    long long bytes;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    ssize_t count;
    if (fd < 0) return -1;
    count = read(fd, value, sizeof(value) - 1);
    {
        int saved_errno = errno;
        close(fd);
        errno = saved_errno;
    }
    if (count <= 0) return -1;
    value[count] = '\0';
    if (strncmp(value, "max", 3) == 0) return -2;
    errno = 0;
    bytes = strtoll(value, &end, 10);
    if (errno != 0 || end == value || bytes < 0) {
        errno = EINVAL;
        return -1;
    }
    if (bytes / MIB_BYTES > LONG_MAX) {
        errno = ERANGE;
        return -1;
    }
    return (long) ((bytes + MIB_BYTES - 1) / MIB_BYTES);
}

int main(int argc, char **argv) {
    pid_t target_pid = 0;
    long warning_mb = 1536;
    long interval_ms = 500;
    char cgroup_dir[PATH_CAP];
    char current_path[PATH_CAP];
    char high_path[PATH_CAP];
    struct timespec delay;
    unsigned long cycle = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--pid") == 0 && i + 1 < argc) {
            target_pid = (pid_t) strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--warning-mb") == 0 && i + 1 < argc) {
            warning_mb = strtol(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--interval-ms") == 0 && i + 1 < argc) {
            interval_ms = strtol(argv[++i], NULL, 10);
        } else {
            fprintf(stderr, "uso: %s --pid PID [--warning-mb 1536] "
                    "[--interval-ms 500]\n", argv[0]);
            return 2;
        }
    }
    if (target_pid <= 1 || warning_mb < 256 || interval_ms < 100) {
        fprintf(stderr, "argumentos invalidos\n");
        return 2;
    }
    if (resolve_cgroup_dir(target_pid, cgroup_dir, sizeof(cgroup_dir)) != 0
            || make_control_path(cgroup_dir, "memory.current", current_path,
                    sizeof(current_path)) != 0
            || make_control_path(cgroup_dir, "memory.high", high_path,
                    sizeof(high_path)) != 0) {
        fprintf(stderr, "nao foi possivel resolver cgroup dedicado: %s\n",
                strerror(errno));
        return 3;
    }
    if (access(current_path, R_OK) != 0 || access(high_path, R_OK) != 0) {
        fprintf(stderr, "telemetria cgroup indisponivel: %s\n", cgroup_dir);
        return 4;
    }

    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    (void) prctl(PR_SET_PDEATHSIG, SIGTERM);
    (void) setpriority(PRIO_PROCESS, 0, 10);
    delay.tv_sec = interval_ms / 1000;
    delay.tv_nsec = (interval_ms % 1000) * 1000000L;
    printf("READY pid=%ld mode=observe warning=%ld path=%s\n",
            (long) target_pid, warning_mb, cgroup_dir);
    fflush(stdout);

    while (keep_running) {
        long available_mb;
        long current_mb;
        long high_mb;
        ++cycle;
        if (kill(target_pid, 0) != 0 && errno == ESRCH) break;
        available_mb = read_mem_available_mb();
        current_mb = read_cgroup_mb(current_path);
        high_mb = read_cgroup_mb(high_path);
        if (available_mb < 0 || current_mb < 0 || high_mb == -1) {
            printf("ERROR read_state=%s\n", strerror(errno));
            fflush(stdout);
            break;
        }
        if (available_mb < warning_mb) {
            printf("PRESSURE available=%ld current=%ld high=", available_mb, current_mb);
            if (high_mb == -2) printf("max\n");
            else printf("%ld\n", high_mb);
            fflush(stdout);
        } else if (cycle == 1 || cycle % 10 == 0) {
            printf("STATUS available=%ld current=%ld high=", available_mb, current_mb);
            if (high_mb == -2) printf("max\n");
            else printf("%ld\n", high_mb);
            fflush(stdout);
        }
        nanosleep(&delay, NULL);
    }
    return 0;
}
